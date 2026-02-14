.PHONY: build deploy clean help

# Variables
REGISTRY = ghcr.io/alignmentresearch
IMAGE_NAME = malign-influence
TARGET ?= devbox
BRANCH_NAME = $(shell git branch --show-current)
IMAGE_TAG = $(TARGET)-$(BRANCH_NAME)
FULL_IMAGE = $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

# Resource configuration (can be overridden)
GPUS ?= 1
CPUS ?= 1
MEMORY ?= 60G
PROJECT_PVC ?= lev-colab

# Kubernetes config
K8S_DEVBOX_TEMPLATE = k8s/devbox.template.yaml
K8S_DEVBOX_YAML_TEMP = k8s/devbox.tmp.yaml
K8S_BUILDER_TEMPLATE = k8s/kinko-builder.template.yaml
K8S_BUILDER_YAML_TEMP = k8s/kinko-builder.tmp.yaml

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'
	@echo ""
	@echo "Resource configuration (override with make deploy CPUS=2 GPUS=4 MEMORY=120G):"
	@echo "  CPUS:     Number of CPUs (default: $(CPUS))"
	@echo "  GPUS:     Number of GPUs (default: $(GPUS))"
	@echo "  MEMORY:   Memory limit (default: $(MEMORY))"

build: ## Build the devbox container using kinko builder
	@# Check that git repository is clean
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Error: Git repository has uncommitted changes. Please commit or stash changes before building."; \
		git status --short; \
		exit 1; \
	fi
	@# Check that we're in sync with remote
	@echo "Checking if branch is in sync with remote..."
	@git fetch origin $(BRANCH_NAME) 2>/dev/null || true
	@LOCAL=$$(git rev-parse @); \
	REMOTE=$$(git rev-parse @{u} 2>/dev/null || echo ""); \
	if [ -n "$$REMOTE" ] && [ "$$LOCAL" != "$$REMOTE" ]; then \
		echo "Error: Branch $(BRANCH_NAME) is not in sync with remote. Please push/pull changes."; \
		echo "Local: $$LOCAL"; \
		echo "Remote: $$REMOTE"; \
		exit 1; \
	fi
	@echo "Git repository is clean and in sync. Proceeding with build..."
	@# Clean up any existing build job
	@kubectl delete job build-image 2>/dev/null || true
	@# Generate kinko builder yaml from template
	@sed -e 's/__BRANCH_NAME__/$(BRANCH_NAME)/g' \
		 -e 's|__IMAGE__|$(FULL_IMAGE)|g' \
	     -e 's/__TARGET__/$(TARGET)/g' \
	     $(K8S_BUILDER_TEMPLATE) > $(K8S_BUILDER_YAML_TEMP)
	@# Apply the build job
	kubectl apply -f $(K8S_BUILDER_YAML_TEMP)
	@# Resume the job
	kubectl patch job build-image --type=merge -p='{"spec":{"suspend":false}}'
	@echo "Kinko build job started. Waiting for pod to be ready..."
	@# Wait for pod to be ready before following logs
	kubectl wait --for=condition=ready --timeout=300s pod -l job-name=build-image
	@echo "Pod ready. Following logs..."
	@# Follow logs until completion
	kubectl logs -f job/build-image
	@# Check if job succeeded (this will fail the make target if build failed)
	kubectl wait --for=condition=complete --timeout=10s job/build-image
	@echo "Build complete: $(FULL_IMAGE)"
	@# Clean up temporary file
	@rm $(K8S_BUILDER_YAML_TEMP)

deploy:
	@kubectl delete job devbox 2>/dev/null || true
	@echo "Deploying devbox with image: $(FULL_IMAGE)"
	@echo "Resource configuration: $(CPUS) CPUs, $(GPUS) GPUs, $(MEMORY) memory"
	@# Generate yaml from template using sed
	@sed -e 's|__IMAGE__|$(FULL_IMAGE)|g' \
	     -e 's|__CPU__|$(CPUS)|g' \
	     -e 's|__GPU__|$(GPUS)|g' \
	     -e 's|__MEMORY__|$(MEMORY)|g' \
	     -e 's|__PROJECT_PVC__|$(PROJECT_PVC)|g' \
	     $(K8S_DEVBOX_TEMPLATE) > $(K8S_DEVBOX_YAML_TEMP)
	@# Apply the configuration
	kubectl apply -f $(K8S_DEVBOX_YAML_TEMP)
	@# Clean up temporary file
	@rm $(K8S_DEVBOX_YAML_TEMP)
	@echo "Deployment complete. Devbox job created with image: $(FULL_IMAGE)"

status: ## Show the status of the devbox job
	@kubectl get job devbox -o wide 2>/dev/null || echo "Devbox job not found"
	@kubectl get pods -l job-name=devbox 2>/dev/null || echo "No devbox pods found"

logs: ## Show logs from the devbox container
	@POD_NAME=$$(kubectl get pods -l job-name=devbox -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
	if [ -n "$$POD_NAME" ]; then \
		kubectl logs $$POD_NAME; \
	else \
		echo "No devbox pods found"; \
	fi

exec: ## Execute a shell in the devbox container
	@POD_NAME=$$(kubectl get pods -l job-name=devbox -o jsonpath='{.items[0].metadata.name}' 2>/dev/null); \
	if [ -n "$$POD_NAME" ]; then \
		kubectl exec -it $$POD_NAME -- /bin/bash; \
	else \
		echo "No running devbox pods found"; \
	fi

delete: ## Delete the devbox job
	@kubectl delete job devbox 2>/dev/null || echo "Devbox job not found"
	@echo "Devbox job deleted"

clean: delete ## Clean up: delete job and remove local images
	@docker rmi $(FULL_IMAGE) 2>/dev/null || echo "Local image not found"
	@echo "Cleanup complete"

info: ## Show current configuration
	@echo "Configuration:"
	@echo "  Branch:     $(BRANCH_NAME)"
	@echo "  Image:      $(FULL_IMAGE)"
	@echo "  Target:     $(TARGET)"
	@echo "  CPUs:       $(CPUS)"
	@echo "  GPUs:       $(GPUS)"
	@echo "  Memory:     $(MEMORY)"
	@echo "  Namespace:  $(NAMESPACE)"
	@echo "  Project PVC: $(PROJECT_PVC)"

# Sweep management targets
sweep-status: ## Show status of sweep jobs (use with SWEEP_ID=xxx)
	@if [ -z "$(SWEEP_ID)" ]; then \
		echo "Usage: make sweep-status SWEEP_ID=<sweep_id>"; \
		echo ""; \
		echo "Active sweeps:"; \
		kubectl get jobs -o wide | grep "sweep-" | head -20; \
	else \
		echo "Status for sweep $(SWEEP_ID):"; \
		kubectl get jobs -l sweep-id=$(SWEEP_ID) -o wide; \
	fi

sweep-logs: ## Get logs from sweep workers (use with SWEEP_ID=xxx)
	@if [ -z "$(SWEEP_ID)" ]; then \
		echo "Usage: make sweep-logs SWEEP_ID=<sweep_id>"; \
		exit 1; \
	fi
	@echo "Logs for sweep $(SWEEP_ID):"
	@kubectl logs -l sweep-id=$(SWEEP_ID) --prefix=true --tail=100

sweep-clean: ## Delete completed sweep jobs (use with SWEEP_ID=xxx)
	@if [ -z "$(SWEEP_ID)" ]; then \
		echo "Usage: make sweep-clean SWEEP_ID=<sweep_id>"; \
		echo ""; \
		echo "WARNING: To clean ALL sweep jobs, use: make sweep-clean-all"; \
		exit 1; \
	fi
	@echo "Deleting jobs for sweep $(SWEEP_ID)..."
	@kubectl delete jobs -l sweep-id=$(SWEEP_ID) 2>/dev/null || echo "No jobs found for sweep $(SWEEP_ID)"

sweep-clean-all: ## Delete ALL sweep jobs (use with caution)
	@echo "WARNING: This will delete ALL sweep jobs!"
	@read -p "Are you sure? (y/N) " confirm && [ "$$confirm" = "y" ] || exit 1
	@kubectl delete jobs -l sweep-id 2>/dev/null || echo "No sweep jobs found"

sweep-list: ## List all active sweeps
	@echo "Active sweep jobs:"
	@kubectl get jobs -L sweep-id,worker-id | grep "sweep-" | head -50