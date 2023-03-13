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

#ifndef XLA_CLIENT_DLPACK_H_
#define XLA_CLIENT_DLPACK_H_

#include <memory>

#include "include/dlpack/dlpack.h"  // from @dlpack
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"

namespace xla {

namespace dlpack {
// If take_ownership is true, ownership of the buffer is handed to DLPack, and
// the receiver may mutate the buffer as they see fit. Otherwise PjRt retains
// ownership of the buffer and it should be immutable.
DLManagedTensor* BufferToDLManagedTensor(std::shared_ptr<xla::PjRtBuffer> buffer,
                                                        bool take_ownership);


}  // namespace dlpack
}  // namespace xla

#endif  // XLA_CLIENT_DLPACK_H_
