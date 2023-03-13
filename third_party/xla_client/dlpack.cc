/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/xla_client/dlpack.h"

#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "include/dlpack/dlpack.h"  // from @dlpack
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"


namespace xla {
namespace {

const char* const kDlTensorCapsuleName = "dltensor";

struct DLPackTensor {
  ~DLPackTensor();

  // `buffer_reference` is populated if we have shared (read-only) access.
  // py::object buffer_reference;

  // `external_reference` is always populated.
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

namespace dlpack {

DLManagedTensor* BufferToDLPackManagedTensor(std::shared_ptr<xla::PjRtBuffer> buffer,
                                                  bool take_ownership) {
  // TF_ASSIGN_OR_RETURN(PyBuffer * buffer, PyBuffer::AsPyBuffer(py_buffer));
  // auto pack = std::make_unique<DLPackTensor>();
  // if (buffer->pjrt_buffer()->on_device_shape().IsTuple()) {
  //   return Unimplemented(
  //       "unsafe_buffer_pointer is not implemented for tuple "
  //       "buffers.");
  // }
  // if (buffer->pjrt_buffer()->on_device_shape().is_dynamic()) {
  //   return Unimplemented("DynamicShape is not implemented in DLPack.");
  // }

  // DLTensor& dt = pack->tensor.dl_tensor;
  // if (take_ownership) {
  //   // Block on outstanding operations, so that it is safe to read or mutate the
  //   // returned buffer.
  //   StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>> buffer_or =
  //       buffer->pjrt_buffer()->ReleaseDeviceMemoryOwnership(
  //           /*wait_for_operations_to_complete=*/true);
  //   if (!buffer_or.ok()) {
  //     return InvalidArgument(
  //         "Buffer synchronization failed converting to DLPack tensor: %s",
  //         buffer_or.status().ToString());
  //   }
  //   pack->external_reference = std::move(buffer_or).value();
  //   if (!pack->external_reference) {
  //     return InvalidArgument(
  //         "Cannot convert deleted/invalid buffer to DLPack tensor.");
  //   }
  // } else {
  //   // Block on outstanding operations, so that it is safe to read or mutate the
  //   // returned buffer.
  //   TF_RETURN_IF_ERROR(buffer->BlockHostUntilReady());
  //   pack->buffer_reference = py::reinterpret_borrow<py::object>(py_buffer);
  //   TF_ASSIGN_OR_RETURN(pack->external_reference,
  //                       buffer->pjrt_buffer()->AcquireExternalReference());
  // }
  // dt.data = pack->external_reference->OpaqueDeviceMemoryDataPointer();
  // pack->tensor.manager_ctx = pack.get();
  // pack->tensor.deleter = DLPackTensorDeleter;
  // TF_ASSIGN_OR_RETURN(dt.device,
  //                     DLDeviceForDevice(*buffer->pjrt_buffer()->device()));
  // dt.device.device_id = buffer->pjrt_buffer()->device()->local_hardware_id();
  // dt.ndim = buffer->pjrt_buffer()->on_device_shape().dimensions_size();
  // TF_ASSIGN_OR_RETURN(
  //     dt.dtype, PrimitiveTypeToDLDataType(
  //                   buffer->pjrt_buffer()->on_device_shape().element_type()));

  // pack->shape = std::vector<int64_t>(
  //     buffer->pjrt_buffer()->on_device_shape().dimensions().begin(),
  //     buffer->pjrt_buffer()->on_device_shape().dimensions().end());
  // pack->strides = StridesForShape(buffer->pjrt_buffer()->on_device_shape());
  // dt.shape = reinterpret_cast<std::int64_t*>(pack->shape.data());
  // dt.strides = reinterpret_cast<std::int64_t*>(pack->strides.data());
  // dt.byte_offset = 0;

  // py::capsule capsule(&pack.release()->tensor, kDlTensorCapsuleName,
  //                     [](PyObject* obj) {
  //                       DLManagedTensor* dlmt = static_cast<DLManagedTensor*>(
  //                           PyCapsule_GetPointer(obj, kDlTensorCapsuleName));
  //                       if (dlmt) {
  //                         DLPackTensorDeleter(dlmt);
  //                       } else {
  //                         // The tensor has been deleted. Clear any error from
  //                         // PyCapsule_GetPointer.
  //                         PyErr_Clear();
  //                       }
  //                     });
  // return capsule;
  return nullptr;
}

xla::ComputationClient::DataPtr DLManagedTensorToBuffer(
    DLManagedTensor* dlmt, std::shared_ptr<PjRtClient> gpu_client){

  // TODO(lyh271596): add client check
  if (dlmt->dl_tensor.ndim < 0) {
    // return InvalidArgument(
    //     "Number of dimensions in DLManagedTensor must be nonnegative, got %d",
    //     dlmt->dl_tensor.ndim);
  }
  TF_ASSERT_OK_AND_ASSIGN(PjRtDevice * device,
                      DeviceForDLDevice( nullptr,
                                        gpu_client.get(),
                                        dlmt->dl_tensor.device));
  absl::Span<int64_t const> dimensions(
      reinterpret_cast<int64_t*>(dlmt->dl_tensor.shape), dlmt->dl_tensor.ndim);
  TF_ASSERT_OK_AND_ASSIGN(PrimitiveType element_type,
                      DLDataTypeToPrimitiveType(dlmt->dl_tensor.dtype));

  std::vector<int64_t> minor_to_major;
  if (dlmt->dl_tensor.strides &&
      absl::c_find(dimensions, 0) == dimensions.end()) {
    absl::Span<int64_t const> strides(
        reinterpret_cast<int64_t*>(dlmt->dl_tensor.strides),
        dlmt->dl_tensor.ndim);
    TF_ASSERT_OK_AND_ASSIGN(minor_to_major, StridesToLayout(dimensions, strides));
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
  std::shared_ptr<xla::PjRtBuffer> buffer =
                      device->client()->CreateViewOfDeviceBuffer(
                          static_cast<char*>(dlmt->dl_tensor.data) +
                              dlmt->dl_tensor.byte_offset,
                          shape, device, on_delete_callback).value();

  xla::ComputationClient::DataPtr data =
        std::make_shared<PjRtData>(device, shape, buffer);

  return data;
}

}  // namespace dlcpack
}  // namespace xla
