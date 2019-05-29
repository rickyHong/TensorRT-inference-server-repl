// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/backends/onnx/autofill.h"

#include <NvInfer.h>
#include "src/backends/onnx/loader.h"
#include "src/backends/onnx/onnx_utils.h"
#include "src/core/autofill.h"
#include "src/core/constants.h"
#include "src/core/filesystem.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"

namespace nvidia { namespace inferenceserver {

class AutoFillOnnxImpl : public AutoFill {
 public:
  AutoFillOnnxImpl(
      const std::string& model_name, const std::string& onnx_filename)
      : AutoFill(model_name), onnx_filename_(onnx_filename)
  {
  }

  Status Fix(ModelConfig* config) override;

  Status SetConfigFromOrtSession(OrtSession* session, OrtAllocator* allocator);

 private:
  Status SetBatchingSupport(OrtSession* session, OrtAllocator* allocator);
  Status SetInputConfig(OrtSession* session, OrtAllocator* allocator);
  Status SetOutputConfig(OrtSession* session, OrtAllocator* allocator);

  const std::string onnx_filename_;
  ModelConfig config_;
};

Status
AutoFillOnnxImpl::Fix(ModelConfig* config)
{
  config->set_platform(kOnnxRuntimeOnnxPlatform);

  // Set name if not already set.
  if (config->name().empty()) {
    config->set_name(model_name_);
  }

  if (config->default_model_filename().empty()) {
    config->set_default_model_filename(onnx_filename_);
  }

  if (config->max_batch_size() == 0) {
    config->set_max_batch_size(config_.max_batch_size());
  }

  // Inputs
  if (config->input().size() == 0) {
    config->mutable_input()->CopyFrom(config_.input());
  }

  // Outputs
  if (config->output().size() == 0) {
    config->mutable_output()->CopyFrom(config_.output());
  }

  return Status::Success;
}

Status
AutoFillOnnxImpl::SetConfigFromOrtSession(OrtSession* session, OrtAllocator* allocator)
{
  RETURN_IF_ERROR(SetBatchingSupport(session, allocator));
  RETURN_IF_ERROR(SetInputConfig(session, allocator));
  RETURN_IF_ERROR(SetOutputConfig(session, allocator));
  return Status::Success;
}

Status
AutoFillOnnxImpl::SetBatchingSupport(OrtSession* session, OrtAllocator* allocator)
{
  config_.set_max_batch_size(0);
  bool support_batching = true;

  // iterate over all input nodes
  size_t num_nodes;
  RETURN_IF_ORT_ERROR(OrtSessionGetInputCount(session, &num_nodes));
  for (size_t i = 0; i < num_nodes; i++) {
    OrtTypeInfo* typeinfo;
    RETURN_IF_ORT_ERROR(OrtSessionGetInputTypeInfo(session, i, &typeinfo));
    const OrtTensorTypeAndShapeInfo* tensor_info =
        OrtCastTypeInfoToTensorInfo(typeinfo);

    size_t num_dims = OrtGetNumOfDimensions(tensor_info);
    std::vector<int64_t> dims(num_dims);
    OrtGetDimensions(tensor_info, (int64_t*)dims.data(), num_dims);
    if ((dims.size() == 0) || (dims[0] != -1)) {
      support_batching = false;
    }
    
    OrtReleaseTypeInfo(typeinfo);
  }

  // iterate over all output nodes
  RETURN_IF_ORT_ERROR(OrtSessionGetOutputCount(session, &num_nodes));
  for (size_t i = 0; i < num_nodes; i++) {
    OrtTypeInfo* typeinfo;
    RETURN_IF_ORT_ERROR(OrtSessionGetOutputTypeInfo(session, i, &typeinfo));
    const OrtTensorTypeAndShapeInfo* tensor_info =
        OrtCastTypeInfoToTensorInfo(typeinfo);

    size_t num_dims = OrtGetNumOfDimensions(tensor_info);
    std::vector<int64_t> dims(num_dims);
    OrtGetDimensions(tensor_info, (int64_t*)dims.data(), num_dims);
    if ((dims.size() == 0) || (dims[0] != -1)) {
      support_batching = false;
    }
    
    OrtReleaseTypeInfo(typeinfo);
  }

  if (support_batching) {
    config_.set_max_batch_size(1);
  }

  return Status::Success;
}

Status
AutoFillOnnxImpl::SetInputConfig(OrtSession* session, OrtAllocator* allocator)
{
  size_t num_nodes;
  RETURN_IF_ORT_ERROR(OrtSessionGetInputCount(session, &num_nodes));

  // iterate over all input nodes
  for (size_t i = 0; i < num_nodes; i++) {
    ModelInput* config_input = config_.add_input();

    char* input_name;
    RETURN_IF_ORT_ERROR(OrtSessionGetInputName(session, i, allocator, &input_name));
    config_input->set_name(input_name);

    OrtTypeInfo* typeinfo;
    RETURN_IF_ORT_ERROR(OrtSessionGetInputTypeInfo(session, i, &typeinfo));
    const OrtTensorTypeAndShapeInfo* tensor_info =
        OrtCastTypeInfoToTensorInfo(typeinfo);
    ONNXTensorElementDataType type = OrtGetTensorElementType(tensor_info);
    config_input->set_data_type(ConvertFromOnnxDataType(type));

    size_t num_dims = OrtGetNumOfDimensions(tensor_info);
    std::vector<int64_t> dims(num_dims);
    OrtGetDimensions(tensor_info, (int64_t*)dims.data(), num_dims);
    // Skip batching dimension
    int dim_idx = (config_.max_batch_size() == 0) ? 0 : 1;
    for (; dim_idx < dims.size(); ++dim_idx) {
      config_input->mutable_dims()->Add(dims[dim_idx]);
    }
    // If input dims are empty then must use a reshape for the
    // input, since 'dims' is not allowed to be empty.
    if (config_input->dims_size() == 0) {
      config_input->mutable_dims()->Add(1);
      config_input->mutable_reshape();
    }
    
    OrtReleaseTypeInfo(typeinfo);
  }

  return Status::Success;
}

Status
AutoFillOnnxImpl::SetOutputConfig(OrtSession* session, OrtAllocator* allocator)
{
  size_t num_nodes;
  RETURN_IF_ORT_ERROR(OrtSessionGetOutputCount(session, &num_nodes));

  // iterate over all output nodes
  for (size_t i = 0; i < num_nodes; i++) {
    ModelOutput* config_output = config_.add_output();

    char* output_name;
    RETURN_IF_ORT_ERROR(OrtSessionGetOutputName(session, i, allocator, &output_name));
    config_output->set_name(output_name);

    OrtTypeInfo* typeinfo;
    RETURN_IF_ORT_ERROR(OrtSessionGetOutputTypeInfo(session, i, &typeinfo));
    const OrtTensorTypeAndShapeInfo* tensor_info =
        OrtCastTypeInfoToTensorInfo(typeinfo);
    ONNXTensorElementDataType type = OrtGetTensorElementType(tensor_info);
    config_output->set_data_type(ConvertFromOnnxDataType(type));

    size_t num_dims = OrtGetNumOfDimensions(tensor_info);
    std::vector<int64_t> dims(num_dims);
    OrtGetDimensions(tensor_info, (int64_t*)dims.data(), num_dims);
    // Skip batching dimension
    int dim_idx = (config_.max_batch_size() == 0) ? 0 : 1;
    for (; dim_idx < dims.size(); ++dim_idx) {
      config_output->mutable_dims()->Add(dims[dim_idx]);
    }
    // If input dims are empty then must use a reshape for the
    // input, since 'dims' is not allowed to be empty.
    if (config_output->dims_size() == 0) {
      config_output->mutable_dims()->Add(1);
      config_output->mutable_reshape();
    }
    
    OrtReleaseTypeInfo(typeinfo);
  }

  return Status::Success;
}

Status
AutoFillOnnx::Create(
    const std::string& model_name, const std::string& model_path,
    std::unique_ptr<AutoFill>* autofill)
{
  std::unique_ptr<AutoFillOnnxImpl> local_autofill;

  std::set<std::string> version_dirs;
  RETURN_IF_ERROR(GetDirectorySubdirs(model_path, &version_dirs));

  if (version_dirs.size() == 0) {
    return Status(
        RequestStatusCode::INTERNAL, "unable to autofill for '" + model_name +
                                         "' due to no version directories");
  }

  OrtSessionOptions* session_options = OrtCreateSessionOptions();
  OrtSetSessionThreadPoolSize(session_options, 1);
  OrtSetSessionGraphOptimizationLevel(session_options, 0);

  OrtSession* session;

  // All versions should share the same model configuration, thus use the first
  // one that can be loaded successfully.
  Status status;
  for (const auto& version : version_dirs) {
    const auto version_path = JoinPath({model_path, version});

    // There must be a single onnx file within the version directory...
    std::set<std::string> onnx_files;
    RETURN_IF_ERROR(GetDirectoryFiles(version_path, &onnx_files));
    if (onnx_files.size() != 1) {
      return Status(
          RequestStatusCode::INTERNAL, "unable to autofill for '" + model_name +
                                           "', unable to find onnx file");
    }

    const std::string onnx_file = *(onnx_files.begin());
    const auto onnx_path = JoinPath({version_path, onnx_file});

    // Load session
    status = OnnxLoader::LoadSession(onnx_path, session_options, &session);

    if (status.IsOk()) {
      local_autofill.reset(new AutoFillOnnxImpl(model_name, onnx_file));
      break;
    }
  }

  OrtReleaseSessionOptions(session_options);
  // Return if none of the version can be loaded successfully
  RETURN_IF_ERROR(status);

  ModelConfig config;
  OrtStatus* ort_status;
  OrtAllocator* allocator;
  ort_status = OrtCreateDefaultAllocator(&allocator);
  if (ort_status == nullptr) {
    status = local_autofill->SetConfigFromOrtSession(session, allocator);
    OrtReleaseAllocator(allocator);
  }
  OnnxLoader::UnloadSession(session);

  RETURN_IF_ORT_ERROR(ort_status);
  RETURN_IF_ERROR(status);

  *autofill = std::move(local_autofill);
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
