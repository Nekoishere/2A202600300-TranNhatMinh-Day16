variable "project_id" {
  description = "GCP Project ID"
  type        = string
  default     = "iron-rex-494202-e9"
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "me-west1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "me-west1-b"
}

variable "hf_token" {
  description = "Hugging Face Token for gated models (like Gemma)"
  type        = string
  sensitive   = true
  default     = "dummy"
}

variable "model_id" {
  description = "Hugging Face Model ID to serve"
  type        = string
  default     = "google/gemma-4-E2B-it"
}

variable "machine_type" {
  description = "GCE Machine Type for the GPU node"
  type        = string
  default     = "n2-standard-8"
}

variable "gpu_type" {
  description = "GPU accelerator type"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  description = "Number of GPUs to attach"
  type        = number
  default     = 1
}
