#include "tensorflow/compiler/xla/xla_client/pjrt_computation_client.h"

#include <algorithm>

#include "absl/algorithm/container.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "include/dlpack/dlpack.h"  // from @dlpack
#include "pjrt_computation_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_api.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_c_api_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_executable.h"
#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/tpu_client.h"
#include "tensorflow/compiler/xla/pjrt/distributed/distributed.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/env_vars.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace xla {

namespace {

std::string PjRtDeviceToString(PjRtDevice* const device) {
  std::string platform =
      absl::AsciiStrToUpper(device->client()->platform_name());
  std::string str = absl::StrFormat("%s:%d", platform, device->id());
  return str;
}

std::vector<std::string> PjRtDevicesToString(
    absl::Span<PjRtDevice* const> devices) {
  std::vector<std::string> strs;
  strs.reserve(devices.size());

  for (auto* device : devices) {
    strs.push_back(PjRtDeviceToString(device));
  }

  return strs;
}

StatusOr<DLDeviceType> DLDeviceTypeForDevice(const PjRtDevice& device) {
  if (device.client()->platform_id() == CpuId()) {
    return kDLCPU;
  } else if (device.client()->platform_id() == GpuId()) {
    const StreamExecutorGpuDevice& gdevice =
        dynamic_cast<const StreamExecutorGpuDevice&>(device);

    if (absl::StrContains(gdevice.device_vendor(), "Advanced Micro Devices")) {
      return kDLROCM;
    } else {
      return kDLCUDA;
    }
  }
  return InvalidArgument("Device %s cannot be used as a DLPack device.",
                         device.DebugString());
}

StatusOr<DLDevice> DLDeviceForDevice(const PjRtDevice& device) {
  DLDevice context;
  TF_ASSIGN_OR_RETURN(context.device_type, DLDeviceTypeForDevice(device));
  context.device_id = device.local_hardware_id();
  return context;
}

// Initializes a distributed runtime client if dist_service_addr is specified
std::shared_ptr<DistributedRuntimeClient> MaybeInitializeDistributedRuntimeClient(
    int local_rank, std::string dist_service_addr) {
  std::shared_ptr<DistributedRuntimeClient> client;
  if (!dist_service_addr.empty()) {
    xla::DistributedRuntimeClient::Options options;
    /* TODO(jonbolin): Use global rank for multi-host setup */
    options.node_id = local_rank;
    client = xla::GetDistributedRuntimeClient(dist_service_addr, options,
        /*use_coordination_service=*/false);
    XLA_CHECK(client->Connect().ok())
        << "Failed to initialize distributed runtime client";
  }
  return std::move(client);
}

struct DLPackTensor {
  ~DLPackTensor();

  // `buffer_reference` is populated if we have shared (read-only) access.
  // py::object buffer_reference;

  // `external_reference` is always populated.
  std::shared_ptr<xla::PjRtBuffer> buffer_reference;
  std::unique_ptr<PjRtBuffer::ExternalReference> external_reference;

  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLManagedTensor tensor;
};

DLPackTensor::~DLPackTensor() {
  // if (buffer_reference) {
  //   GlobalPyRefManager()->AddGarbage(
  //       absl::MakeSpan(&buffer_reference, /*size=*/1));
  // }
}

void DLPackTensorDeleter(DLManagedTensor* t) {
  if (t) {
    delete static_cast<DLPackTensor*>(t->manager_ctx);
  }
}

StatusOr<DLDataType> PrimitiveTypeToDLDataType(PrimitiveType type) {
  switch (type) {
    case S8:
      return DLDataType{kDLInt, 8, 1};
    case S16:
      return DLDataType{kDLInt, 16, 1};
    case S32:
      return DLDataType{kDLInt, 32, 1};
    case S64:
      return DLDataType{kDLInt, 64, 1};
    case U8:
      return DLDataType{kDLUInt, 8, 1};
    case U16:
      return DLDataType{kDLUInt, 16, 1};
    case U32:
      return DLDataType{kDLUInt, 32, 1};
    case U64:
      return DLDataType{kDLUInt, 64, 1};
    case F16:
      return DLDataType{kDLFloat, 16, 1};
    case F32:
      return DLDataType{kDLFloat, 32, 1};
    case F64:
      return DLDataType{kDLFloat, 64, 1};
    case BF16:
      return DLDataType{kDLBfloat, 16, 1};
    case PRED:
      return DLDataType{kDLUInt, 8, 1};
    case C64:
      return DLDataType{kDLComplex, 64, 1};
    case C128:
      return DLDataType{kDLComplex, 128, 1};
    default:
      return Unimplemented("XLA type %s has no DLPack equivalent",
                           PrimitiveType_Name(type));
  }
}

StatusOr<PrimitiveType> DLDataTypeToPrimitiveType(DLDataType type) {
  if (type.lanes != 1) {
    return Unimplemented("DLPack types with lanes != 1 not implemented, got %d",
                         type.lanes);
  }
  switch (type.code) {
    case kDLInt:
      switch (type.bits) {
        case 8:
          return S8;
        case 16:
          return S16;
        case 32:
          return S32;
        case 64:
          return S64;
        default:
          return Unimplemented(
              "Invalid or unsupported DLPack integer width: %d bits",
              type.bits);
      }
    case kDLUInt:
      switch (type.bits) {
        case 8:
          return U8;
        case 16:
          return U16;
        case 32:
          return U32;
        case 64:
          return U64;
        default:
          return Unimplemented(
              "Invalid or unsupported DLPack unsigned integer width: %d bits",
              type.bits);
      }
    case kDLFloat:
      switch (type.bits) {
        case 16:
          return F16;
        case 32:
          return F32;
        case 64:
          return F64;
        default:
          return Unimplemented(
              "Invalid or unsupported DLPack float width: %d bits", type.bits);
      }
    case kDLBfloat:
      switch (type.bits) {
        case 16:
          return BF16;
        default:
          return Unimplemented(
              "Invalid or unsupported DLPack Bfloat width: %d bits", type.bits);
      }
    case kDLComplex:
      switch (type.bits) {
        case 64:
          return C64;
        case 128:
          return C128;
        default:
          return Unimplemented(
              "Invalid or unsupported DLPack complex width: %d bits",
              type.bits);
      }
    default:
      return Unimplemented("Unknown or invalid DLPack type code %d", type.code);
  }
}

// Returns the strides for `shape`.
std::vector<int64_t> StridesForShape(const Shape& shape) {
  std::vector<int64_t> strides;
  CHECK(shape.IsArray());
  CHECK(shape.has_layout());

  strides.resize(shape.dimensions_size());
  int64_t stride = 1;
  for (int i : shape.layout().minor_to_major()) {
    strides.at(i) = stride;
    stride *= shape.dimensions(i);
  }
  return strides;
}

StatusOr<std::vector<int64_t>> StridesToLayout(
    absl::Span<int64_t const> dims, absl::Span<int64_t const> strides) {
  CHECK_EQ(dims.size(), strides.size());
  std::vector<int64_t> minor_to_major(dims.size());
  std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
  absl::c_sort(minor_to_major, [&](int a, int b) {
    if (strides[a] < strides[b]) {
      return true;
    }
    if (strides[a] > strides[b]) {
      return false;
    }
    // If two dimensions have the same stride, prefer the major-to-minor
    // interpretation of the ordering, since that's what JAX wants.
    return b < a;
  });
  int64_t stride = 1;
  for (int64_t d : minor_to_major) {
    if (dims[d] > 1 && strides[d] != stride) {
      return Unimplemented(
          "Only DLPack tensors with trivial (compact) striding are supported; "
          "i.e., tensors whose striding represents a transposition of the "
          "underlying buffer but not broadcasting. Dimensions were: [%s], "
          "strides were [%s].",
          absl::StrJoin(dims, ","), absl::StrJoin(strides, ","));
    }
    stride *= dims[d];
  }
  return minor_to_major;
}

StatusOr<PjRtDevice*> DeviceForDLDevice(const PjRtClient* cpu_client,
                                        const PjRtClient* gpu_client,
                                        const DLDevice& context) {
  switch (context.device_type) {
    case kDLCPU:
      if (cpu_client == nullptr) {
        return InvalidArgument(
            "DLPack tensor is on CPU, but no CPU backend was provided.");
      }
      TF_RET_CHECK(cpu_client->platform_id() == CpuId());
      return cpu_client->LookupAddressableDevice(context.device_id);
    case kDLCUDA:
      if (gpu_client == nullptr) {
        return InvalidArgument(
            "DLPack tensor is on GPU, but no GPU backend was provided.");
      }
      TF_RET_CHECK(gpu_client->platform_id() == GpuId());
      return gpu_client->LookupAddressableDevice(context.device_id);
    case kDLROCM:
      if (gpu_client == nullptr) {
        return InvalidArgument(
            "DLPack tensor is on GPU, but no GPU backend was provided.");
      }
      TF_RET_CHECK(gpu_client->platform_id() == GpuId());
      return gpu_client->LookupAddressableDevice(context.device_id);
    default:
      return InvalidArgument("Unknown/unsupported DLPack device type %d",
                             context.device_type);
  }
}

}  // namespace

PjRtComputationClient::PjRtComputationClient() {
  std::string device_type = sys_util::GetEnvString(env::kEnvPjRtDevice, "");
  if (device_type == "CPU") {
    TF_VLOG(1) << "Initializing PjRt CPU client...";
    bool async = sys_util::GetEnvBool(env::kEnvPjrtAsyncCpuClient, true);
    int cpu_device_count = sys_util::GetEnvInt(env::kEnvNumCpu, 1);
    client_ = std::move(xla::GetTfrtCpuClient(async, cpu_device_count).value());
  } else if (device_type == "TPU" || device_type == "TPU_C_API") {
    TF_VLOG(1) << "Initializing TFRT TPU client...";
    XLA_CHECK_OK(pjrt::LoadPjrtPlugin(
        "tpu", sys_util::GetEnvString(env::kEnvTpuLibraryPath, "libtpu.so")));
    supports_logical_on_device_shape_ = false;
    client_ = std::move(xla::GetCApiClient("TPU").value());
  } else if (device_type == "TPU_LEGACY") {
    TF_VLOG(1) << "Initializing PjRt StreamExecutor TPU client...";
    int64_t max_inflight_computations = sys_util::GetEnvInt(
        env::kEnvPjRtTpuMaxInflightComputations, /*defval=*/32);
    client_ = xla::GetTpuClient(max_inflight_computations).value();
  } else if (device_type == "GPU") {
    TF_VLOG(1) << "Initializing PjRt GPU client...";
    bool async = sys_util::GetEnvBool(env::kEnvPjrtAsyncGpuClient, true);
    int local_rank = sys_util::GetEnvInt(env::kEnvPjRtLocalRank, 0);
    std::string dist_service_addr =
        sys_util::GetEnvString(env::kEnvPjrtDistServiceAddr, "");
    auto distributed_client = MaybeInitializeDistributedRuntimeClient(
        local_rank, dist_service_addr);
    auto allowed_devices = std::make_optional<std::set<int>>(std::set{local_rank});
    client_ = std::move(xla::GetStreamExecutorGpuClient(
                  /*asynchronous=*/async, GpuAllocatorConfig{},
                  /*distributed_client=*/distributed_client, /*node_id=*/local_rank,
                  allowed_devices = allowed_devices)
                  .value());
  } else {
    XLA_ERROR() << absl::StrFormat("Unknown %s '%s'", env::kEnvPjRtDevice,
                                   device_type);
  }

  XLA_CHECK(client_.get() != nullptr);

  for (auto* device : client_->devices()) {
    std::string device_str = PjRtDeviceToString(device);
    string_to_device_.emplace(device_str, device);
  }
}

void PjRtComputationClient::PjRtData::Assign(const Data& data) {
  const PjRtData& pjrt_data = dynamic_cast<const PjRtData&>(data);
  if (&pjrt_data != this) {
    buffer = pjrt_data.buffer;
  }
}

ComputationClient::DataPtr PjRtComputationClient::CreateDataPlaceholder(
    std::string device, Shape shape) {
  return std::make_shared<PjRtData>(device, shape);
}

std::vector<ComputationClient::DataPtr> PjRtComputationClient::GetDataShards(
    ComputationClient::DataPtr data) {
  std::vector<ComputationClient::DataPtr> shards;
  if (PjRtShardedData* sharded_data =
          dynamic_cast<PjRtShardedData*>(data.get())) {
    for (auto shard : sharded_data->shards) {
      shards.push_back(std::make_shared<PjRtData>(
          shard->device(), shard->shape(), shard->buffer));
    }
  } else {
    shards.push_back(data);
  }
  return shards;
}

ComputationClient::DataPtr PjRtComputationClient::GetUninitializedData(
    const std::string& device, Shape shape) {
  PjRtDevice* pjrt_device = StringToPjRtDevice(device);
  std::shared_ptr<xla::PjRtBuffer> buffer =
      std::move(client_->CreateUninitializedBuffer(shape, pjrt_device).value());
  ComputationClient::DataPtr data =
      std::make_shared<PjRtData>(device, shape, buffer);
  return data;
}

ComputationClient::DataPtr PjRtComputationClient::CreateViewOfDeviceBuffer(
    void* device_ptr, const Shape& shape, const std::string& device,
    std::function<void()> on_delete_callback) {
  PjRtDevice* pjrt_device = StringToPjRtDevice(device);
  std::shared_ptr<xla::PjRtBuffer> buffer =
      std::move(client_
                    ->CreateViewOfDeviceBuffer(device_ptr, shape, pjrt_device,
                                               on_delete_callback)
                    .value());
  ComputationClient::DataPtr data =
      std::make_shared<PjRtData>(device, shape, buffer);
  return data;
}

ComputationClient::DataPtr PjRtComputationClient::CreateViewOfDeviceBuffer(
    DLManagedTensor* dlmt) {
  return DLManagedTensorToBuffer(dlmt, client_);
}

DLManagedTensor* PjRtComputationClient::GetDLManagedTensor(
    ComputationClient::DataPtr data) {
  // TF_ASSIGN_OR_RETURN(PyBuffer * buffer, PyBuffer::AsPyBuffer(py_buffer));

  bool take_ownership = false;
  auto buffer = dynamic_cast<const PjRtData&>(*data).buffer;
  DLPackTensor* pack(new DLPackTensor);
  if (buffer->on_device_shape().IsTuple()) {
    // return Unimplemented(
    //     "unsafe_buffer_pointer is not implemented for tuple "
    //     "buffers.");
  }
  if (buffer->on_device_shape().is_dynamic()) {
    // return Unimplemented("DynamicShape is not implemented in DLPack.");
  }

  DLTensor& dt = pack->tensor.dl_tensor;
  if (take_ownership) {
    // Block on outstanding operations, so that it is safe to read or mutate the
    // returned buffer.
    // StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>> buffer_or =
    //     buffer->pjrt_buffer()->ReleaseDeviceMemoryOwnership(
    //         /*wait_for_operations_to_complete=*/true);
    // if (!buffer_or.ok()) {
    //   return InvalidArgument(
    //       "Buffer synchronization failed converting to DLPack tensor: %s",
    //       buffer_or.status().ToString());
    // }
    // pack->external_reference = std::move(buffer_or).value();
    // if (!pack->external_reference) {
    //   return InvalidArgument(
    //       "Cannot convert deleted/invalid buffer to DLPack tensor.");
    // }
  } else {
    // Block on outstanding operations, so that it is safe to read or mutate the
    // returned buffer.
    // TF_RETURN_IF_ERROR(buffer->BlockHostUntilReady());
    // pack->buffer_reference = py::reinterpret_borrow<py::object>(py_buffer);
    pack->buffer_reference = buffer;
    pack->external_reference = buffer->AcquireExternalReference().value();
  }
  dt.data = pack->external_reference->OpaqueDeviceMemoryDataPointer();
  pack->tensor.manager_ctx = pack;
  pack->tensor.deleter = DLPackTensorDeleter;
  dt.device = DLDeviceForDevice(*buffer->device()).value();
  dt.device.device_id = buffer->device()->local_hardware_id();
  dt.ndim = buffer->on_device_shape().dimensions_size();

  dt.dtype = PrimitiveTypeToDLDataType(buffer->on_device_shape().element_type())
                 .value();

  pack->shape =
      std::vector<int64_t>(buffer->on_device_shape().dimensions().begin(),
                           buffer->on_device_shape().dimensions().end());
  pack->strides = StridesForShape(buffer->on_device_shape());
  dt.shape = reinterpret_cast<std::int64_t*>(pack->shape.data());
  dt.strides = reinterpret_cast<std::int64_t*>(pack->strides.data());
  dt.byte_offset = 0;

  // py::capsule capsule(&pack.release()->tensor, kDlTensorCapsuleName,
  //                     [](PyObject* obj) {
  //                       DLManagedTensor* dlmt =
  //                       static_cast<DLManagedTensor*>(
  //                           PyCapsule_GetPointer(obj, kDlTensorCapsuleName));
  //                       if (dlmt) {
  //                         DLPackTensorDeleter(dlmt);
  //                       } else {
  //                         // The tensor has been deleted. Clear any error
  //                         from
  //                         // PyCapsule_GetPointer.
  //                         PyErr_Clear();
  //                       }
  //                     });
  // return capsule;
  return &(pack->tensor);
}

std::vector<ComputationClient::DataPtr> PjRtComputationClient::TransferToServer(
    absl::Span<const TensorSource> tensors) {
  metrics::TimedSection timed(TransferToServerMetric());
  tensorflow::profiler::TraceMe activity(
      "PjRtComputationClient::TransferToServer",
      tensorflow::profiler::TraceMeLevel::kInfo);
  std::vector<ComputationClient::DataPtr> datas;
  datas.reserve(tensors.size());
  int64_t total_size = 0;
  for (auto& tensor : tensors) {
    PjRtDevice* pjrt_device = StringToPjRtDevice(tensor.device);

    auto literal = std::make_shared<xla::Literal>(tensor.shape);
    tensor.populate_fn(tensor, literal->untyped_data(), literal->size_bytes());
    std::vector<int64_t> byte_strides(literal->shape().dimensions_size());
    XLA_CHECK_OK(
        ShapeUtil::ByteStrides(literal->shape(), absl::MakeSpan(byte_strides)));
    total_size += literal->size_bytes();

    // Avoid use-after-free on `literal` due to unsequenced move and use.
    xla::Literal* literal_pointer = literal.get();
    std::shared_ptr<xla::PjRtBuffer> buffer = std::move(
        client_
            ->BufferFromHostBuffer(
                literal_pointer->untyped_data(),
                literal_pointer->shape().element_type(),
                literal_pointer->shape().dimensions(), byte_strides,
                xla::PjRtClient::HostBufferSemantics::
                    kImmutableUntilTransferCompletes,
                [literal{std::move(literal)}]() { /* frees literal */ },
                pjrt_device)
            .value());

    ComputationClient::DataPtr data =
        std::make_shared<PjRtData>(tensor.device, tensor.shape, buffer);
    datas.push_back(data);
  }
  OutboundDataMetric()->AddSample(total_size);
  CreateDataHandlesCounter()->AddValue(datas.size());

  return datas;
}

ComputationClient::DataPtr PjRtComputationClient::TransferShardsToServer(
    absl::Span<const TensorSource> tensor_shards, std::string device,
    xla::Shape shape, xla::OpSharding sharding) {
  TF_VLOG(1) << "TransferShardsToServer with " << tensor_shards.size()
             << " shards.";
  auto data_shards = TransferToServer(tensor_shards);
  std::vector<std::shared_ptr<PjRtData>> pjrt_data_shards;
  for (auto& shard : data_shards) {
    auto pjrt_shard = dynamic_cast<PjRtData*>(shard.get());
    pjrt_data_shards.push_back(std::make_shared<PjRtData>(
        pjrt_shard->device(), pjrt_shard->shape(), pjrt_shard->buffer));
  }
  return std::make_shared<PjRtShardedData>(device, shape, pjrt_data_shards,
                                           sharding);
}

ComputationClient::DataPtr PjRtComputationClient::CopyToDevice(
    ComputationClient::DataPtr data, std::string dst) {
  tensorflow::profiler::TraceMe activity(
      "PjRtComputationClient::CopyToDevice",
      tensorflow::profiler::TraceMeLevel::kInfo);
  const PjRtData* pjrt_data = dynamic_cast<PjRtData*>(data.get());
  XLA_CHECK(pjrt_data->HasValue()) << "Can't copy invalid device data.";

  PjRtDevice* dst_device = StringToPjRtDevice(dst);
  XLA_CHECK(dst_device->IsAddressable()) << dst << "is not addressable.";

  // Returns error if the buffer is already on `dst_device`.
  StatusOr<std::unique_ptr<PjRtBuffer>> status_or =
      pjrt_data->buffer->CopyToDevice(dst_device);
  XLA_CHECK(status_or.ok())
      << pjrt_data->device() << " buffer already exists on " << dst;

  return std::make_shared<PjRtData>(dst, pjrt_data->shape(),
                                    std::move(status_or.value()));
}

ComputationClient::DataPtr PjRtComputationClient::ReplicateShardedData(
    const ComputationClient::DataPtr& handle) {
  if (PjRtShardedData* sharded_data =
          dynamic_cast<PjRtShardedData*>(handle.get())) {
    TF_VLOG(1) << "ReplicateShardedData (handle=" << handle->GetOpaqueHandle()
               << ", shape=" << handle->shape() << ")";
    xla::XlaBuilder b("ReplicateShardedData");
    xla::Shape shape = sharded_data->shape();
    b.SetSharding(sharded_data->GetSharding());

    // perform a simple identity calculation to reassemble the input as
    // replicated output.
    auto x = xla::Parameter(&b, 0, shape, "p0");
    b.SetSharding(xla::HloSharding::Replicate().ToProto());
    auto y = xla::Div(x, ConstantR0<float>(&b, 2));
    auto z = xla::Add(y, y);

    xla::XlaComputation computation =
        ConsumeValue(b.Build(/*remove_dynamic_dimensions=*/false));
    xla::ProgramShape program_shape =
        ConsumeValue(computation.GetProgramShape());

    std::string device = GetDefaultDevice();
    std::vector<xla::ComputationClient::CompileInstance> instances;
    instances.push_back({std::move(computation), device,
                         GetCompilationDevices(device, {}), &shape,
                         /*should_wrap_parameter=*/false,
                         /*is_sharded=*/true});
    std::vector<std::shared_ptr<xla::ComputationClient::Computation>>
        computations = Compile(std::move(instances));

    auto shards = sharded_data->shards;
    XLA_CHECK_EQ(shards.size(), GetLocalDevices().size());
    std::vector<std::vector<ComputationClient::DataPtr>> arguments_by_device(
        GetLocalDevices().size(), std::vector<ComputationClient::DataPtr>(1));
    for (auto shard : shards) {
      std::vector<std::string> device_spec =
          absl::StrSplit(shard->device(), ':');
      XLA_CHECK_EQ(device_spec.size(), 2)
          << "Invalid device specification: " << shard->device();
      int device_i = std::stoi(device_spec[1]);
      arguments_by_device[device_i][0] = shard;
    }
    xla::ComputationClient::ExecuteReplicatedOptions execute_options;
    return ExecuteReplicated(*computations.front(), arguments_by_device,
                             GetLocalDevices(), execute_options)[0][0];
  }
  return handle;
}

std::vector<xla::Literal> PjRtComputationClient::TransferFromServer(
    absl::Span<const DataPtr> handles) {
  metrics::TimedSection timed(TransferFromServerMetric());
  tensorflow::profiler::TraceMe activity(
      "PjRtComputationClient::TransferFromServer",
      tensorflow::profiler::TraceMeLevel::kInfo);
  std::vector<xla::Literal> literals;
  literals.reserve(handles.size());

  int64_t total_size = 0;
  for (auto handle : handles) {
    // Use XLA replication to reassemble the sharded data. If input handle
    // is not sharded, then it is a no-op.
    auto new_handle = ReplicateShardedData(handle);
    const PjRtData& pjrt_data = dynamic_cast<const PjRtData&>(*new_handle);

    // TODO(wcromar): Only use logical_on_device_shape when PJRT C API supports
    // it.
    xla::Shape target_shape = ShapeUtil::DeviceShapeToHostShape(
        supports_logical_on_device_shape_
            ? pjrt_data.buffer->logical_on_device_shape().value()
            : pjrt_data.buffer->on_device_shape());
    auto& literal = literals.emplace_back(target_shape);

    // PJRT will always try to copy the full bounded size into our literal. If
    // the bounded size is larger than the logical output size, we have to
    // allocate a bounded-size literal and copy a slice of the values into our
    // output literal.
    if (pjrt_data.buffer->on_device_shape().is_static()) {
      XLA_CHECK_OK(pjrt_data.buffer->ToLiteralSync(&literal));
    } else {
      std::shared_ptr<xla::Literal> bounded_literal =
          pjrt_data.buffer->ToLiteralSync().value();
      XLA_CHECK_OK(literal.CopySliceFrom(
          *bounded_literal,
          /*src_base=*/std::vector<int64_t>(target_shape.rank(), 0),
          /*dest_base=*/std::vector<int64_t>(target_shape.rank(), 0),
          /*copy_size=*/target_shape.dimensions()));
    }
    total_size += literal.size_bytes();
  }
  InboundDataMetric()->AddSample(total_size);

  return literals;
}

std::vector<ComputationClient::ComputationPtr> PjRtComputationClient::Compile(
    std::vector<ComputationClient::CompileInstance> instances) {
  metrics::TimedSection timed(CompileMetric());
  tensorflow::profiler::TraceMe activity(
      "PjRtComputationClient::Compile",
      tensorflow::profiler::TraceMeLevel::kInfo);
  std::vector<ComputationClient::ComputationPtr> computations;

  for (auto& instance : instances) {
    xla::CompileOptions compile_options;
    if (instance.is_sharded) {
      // TODO(yeounoh) multi-host, multi-slice configurations
      compile_options.executable_build_options.set_use_spmd_partitioning(true);
      compile_options.executable_build_options.set_num_partitions(
          client_->device_count());
      compile_options.executable_build_options.set_num_replicas(1);
      compile_options.parameter_is_tupled_arguments =
          instance.parameter_is_tupled_arguments;

      // TODO(244391366) verify this is correct for the collectives ops
      xla::DeviceAssignment device_assignment(1, client_->device_count());
      device_assignment.FillIota(0);
      compile_options.executable_build_options.set_device_assignment(
          device_assignment);
    } else {
      // TODO(wcromar): set compile_options.argument_layouts, enable strict
      // shapes
      compile_options.executable_build_options.set_num_partitions(1);
      compile_options.executable_build_options.set_num_replicas(
          client_->device_count());
      compile_options.parameter_is_tupled_arguments =
          instance.parameter_is_tupled_arguments;

      xla::DeviceAssignment device_assignment(client_->device_count(), 1);
      device_assignment.FillIota(0);
      compile_options.executable_build_options.set_device_assignment(
          device_assignment);
    }

    PjRtDevice* pjrt_device = StringToPjRtDevice(instance.compilation_device);
    std::unique_ptr<xla::PjRtLoadedExecutable> executable =
        ConsumeValue(client_->Compile(instance.computation, compile_options));

    const auto& hlo_modules = ConsumeValue(executable->GetHloModules());
    HloComputation* hlo_computation = hlo_modules[0]->entry_computation();
    xla::ProgramShape program_shape =
        xla::ProgramShape(hlo_computation->ToProto().program_shape());

    std::shared_ptr<PjRtComputation> pjrt_computation =
        std::make_shared<PjRtComputation>(
            std::move(xla::XlaComputation(hlo_modules[0]->ToProto())),
            program_shape, instance.devices, std::move(executable));

    computations.push_back(pjrt_computation);

    CreateCompileHandlesCounter()->AddValue(1);
  }

  return computations;
}

std::vector<ComputationClient::DataPtr>
PjRtComputationClient::ExecuteComputation(
    const ComputationClient::Computation& computation,
    absl::Span<const ComputationClient::DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  // Shared ownership of the timed section ensures that it will only get logged
  // once both `ExecuteComputation` and the async work in `ExecuteSharded` are
  // complete; a copy is held from the lambda that releases it when done.
  auto timed = std::make_shared<metrics::TimedSection>(ExecuteMetric());
  tensorflow::profiler::TraceMe activity(
      "PjRtComputationClient::ExecuteComputation",
      tensorflow::profiler::TraceMeLevel::kInfo);
  TF_VLOG(1) << "Executing PjRt computation on " << device;
  const PjRtComputation& pjrt_computation =
      dynamic_cast<const PjRtComputation&>(computation);

  xla::PjRtDevice* pjrt_device = StringToPjRtDevice(device);
  XLA_CHECK(pjrt_device->IsAddressable()) << pjrt_device->DebugString();

  std::vector<xla::PjRtBuffer*> buffers;
  buffers.reserve(arguments.size());
  for (auto& argument : arguments) {
    const PjRtData* pjrt_data = dynamic_cast<PjRtData*>(argument.get());

    XLA_CHECK(pjrt_device == pjrt_data->buffer->device())
        << pjrt_device->DebugString() << " vs "
        << pjrt_data->buffer->device()->DebugString();
    buffers.push_back(pjrt_data->buffer.get());
  }

  xla::ExecuteOptions execute_options;
  execute_options.untuple_result = options.explode_tuple;
  execute_options.strict_shape_checking = false;

  std::optional<PjRtFuture<Status>> returned_future;
  std::vector<std::unique_ptr<xla::PjRtBuffer>> results =
      pjrt_computation.executable
          ->ExecuteSharded(buffers, pjrt_device, execute_options,
                           returned_future)
          .value();

  // Signal that `ExecuteSharded` has completed for the ExecuteTime metric.
  // Copies the `timed` shared pointer into the lambda.
  returned_future->OnReady([timed](Status unused) mutable { timed.reset(); });

  std::vector<DataPtr> datas;
  datas.reserve(results.size());
  for (auto& result : results) {
    std::unique_ptr<xla::PjRtBuffer> buffer = std::move(result);

    std::shared_ptr<PjRtData> data = std::make_shared<PjRtData>(
        device, buffer->on_device_shape(), std::move(buffer));

    datas.push_back(data);
  }
  CreateDataHandlesCounter()->AddValue(datas.size());

  TF_VLOG(1) << "Returning " << datas.size() << " results";
  return datas;
}

std::vector<std::vector<ComputationClient::DataPtr>>
PjRtComputationClient::ExecuteReplicated(
    const ComputationClient::Computation& computation,
    const std::vector<std::vector<ComputationClient::DataPtr>>& arguments,
    absl::Span<const std::string> devices,
    const ExecuteReplicatedOptions& options) {
  const PjRtComputation& pjrt_computation =
      dynamic_cast<const PjRtComputation&>(computation);
  XLA_CHECK(devices.size() == arguments.size())
      << "ExecuteReplicated over " << devices.size() << " devices, but "
      << arguments.size() << " arguments devices.";

  std::vector<std::vector<PjRtBuffer*>> argument_handles;
  for (int32_t i = 0; i < devices.size(); ++i) {
    xla::PjRtDevice* pjrt_device = StringToPjRtDevice(devices[i]);
    XLA_CHECK(pjrt_device->IsAddressable()) << pjrt_device->DebugString();

    std::vector<PjRtBuffer*> buffers;
    for (auto& argument : arguments[i]) {
      const PjRtData* pjrt_data = dynamic_cast<PjRtData*>(argument.get());

      XLA_CHECK(pjrt_device == pjrt_data->buffer->device())
          << pjrt_device->DebugString() << " vs "
          << pjrt_data->buffer->device()->DebugString();
      buffers.push_back(pjrt_data->buffer.get());
    }
    argument_handles.push_back(buffers);
  }

  xla::ExecuteOptions execute_options;
  execute_options.untuple_result = options.explode_tuple;
  execute_options.strict_shape_checking = true;
  // TODO(yeounoh) currently only support single-slice execution
  execute_options.multi_slice_config = nullptr;
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results =
      pjrt_computation.executable->Execute(argument_handles, execute_options)
          .value();

  std::vector<std::vector<ComputationClient::DataPtr>> data_handles;
  data_handles.reserve(results.size());
  for (int32_t i = 0; i < results.size(); ++i) {
    xla::PjRtDevice* pjrt_device = StringToPjRtDevice(devices[i]);
    XLA_CHECK(pjrt_device->IsAddressable())
        << pjrt_device->DebugString() << " is not addressable.";

    std::vector<ComputationClient::DataPtr> datas;
    datas.reserve(results[i].size());
    for (int32_t j = 0; j < results[i].size(); ++j) {
      std::unique_ptr<xla::PjRtBuffer> buffer = std::move(results[i][j]);
      XLA_CHECK(pjrt_device == buffer->device())
          << "Exepcted device: " << pjrt_device->DebugString()
          << " vs. actual device: " << buffer->device()->DebugString();

      std::shared_ptr<PjRtData> data = std::make_shared<PjRtData>(
          devices[i], buffer->on_device_shape(), std::move(buffer));
      datas.push_back(data);
    }
    data_handles.push_back(datas);
  }

  TF_VLOG(1) << "Returning " << data_handles.size() << " sets of results";
  return data_handles;
}

size_t PjRtComputationClient::GetNumDevices() const {
  return client_->addressable_device_count();
}

std::string PjRtComputationClient::GetDefaultDevice() const {
  return PjRtDeviceToString(client_->addressable_devices()[0]);
}

std::vector<std::string> PjRtComputationClient::GetLocalDevices() const {
  return PjRtDevicesToString(client_->addressable_devices());
}

std::vector<std::string> PjRtComputationClient::GetAllDevices() const {
  return PjRtDevicesToString(client_->devices());
}

int PjRtComputationClient::GetNumProcesses() const {
  int max_process_index = client_->process_index();
  for (auto* device : client_->devices()) {
    max_process_index = std::max(max_process_index, device->process_index());
  }

  return max_process_index + 1;
};

const absl::flat_hash_map<std::string, xla::ComputationClient::DeviceAttribute>&
PjRtComputationClient::GetDeviceAttributes(const std::string& device) {
  return PjRtComputationClient::StringToPjRtDevice(device)->Attributes();
}

void PjRtComputationClient::SetReplicationDevices(
    std::shared_ptr<std::vector<std::string>> devices) {
  replication_devices_ = std::move(devices);
}

std::shared_ptr<std::vector<std::string>>
PjRtComputationClient::GetReplicationDevices() {
  return replication_devices_;
}

ComputationClient::DataPtr PjRtComputationClient::DLManagedTensorToBuffer(
    DLManagedTensor* dlmt, std::shared_ptr<PjRtClient> gpu_client) {
  // TODO(lyh271596): add client check
  if (dlmt->dl_tensor.ndim < 0) {
    // return InvalidArgument(
    //     "Number of dimensions in DLManagedTensor must be nonnegative, got
    //     %d", dlmt->dl_tensor.ndim);
  }
  PjRtDevice* device =
      DeviceForDLDevice(nullptr, gpu_client.get(), dlmt->dl_tensor.device)
          .value();
  absl::Span<int64_t const> dimensions(
      reinterpret_cast<int64_t*>(dlmt->dl_tensor.shape), dlmt->dl_tensor.ndim);
  PrimitiveType element_type =
      DLDataTypeToPrimitiveType(dlmt->dl_tensor.dtype).value();

  std::vector<int64_t> minor_to_major;
  if (dlmt->dl_tensor.strides &&
      absl::c_find(dimensions, 0) == dimensions.end()) {
    absl::Span<int64_t const> strides(
        reinterpret_cast<int64_t*>(dlmt->dl_tensor.strides),
        dlmt->dl_tensor.ndim);
    minor_to_major = StridesToLayout(dimensions, strides).value();
  } else {
    minor_to_major.resize(dlmt->dl_tensor.ndim);
    std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);
  }
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(element_type, dimensions,
                                                    minor_to_major);

  std::function<void()> on_delete_callback;
  if (dlmt->deleter) {
    on_delete_callback = [dlmt]() { dlmt->deleter(dlmt); };
  }
  auto buffer =
      device->client()
          ->CreateViewOfDeviceBuffer(static_cast<char*>(dlmt->dl_tensor.data) +
                                         dlmt->dl_tensor.byte_offset,
                                     shape, device, on_delete_callback)
          .value();

  ComputationClient::DataPtr data = std::make_shared<PjRtData>(
      PjRtDeviceToString(device), shape, std::move(buffer));

  return data;
}

xla::PjRtDevice* PjRtComputationClient::StringToPjRtDevice(
    const std::string& device) {
  XLA_CHECK(string_to_device_.find(device) != string_to_device_.end())
      << "Unknown device " << device;
  xla::PjRtDevice* pjrt_device = string_to_device_[device];
  return pjrt_device;
}

std::map<std::string, Metric> PjRtComputationClient::GetMetrics() const {
  // TODO(jonbolin): Add any PJRt-client-specific metrics here
  return {};
}

}  // namespace xla
