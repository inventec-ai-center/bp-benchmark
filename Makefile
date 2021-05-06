#!make
include envfile

SHELL=/bin/bash #/bin/bash


PROCESS_IMAGE := $(IMAGE)-process
TRAIN_IMAGE := $(IMAGE)-train
LATEST = latest


clean:
	@{ \
        set -euo pipefail ;\
        IMG=$$(docker images -q "$(REGISTRY)/$(NAMESPACE)/$(TRAIN_IMAGE):$(VERSION)") ;\
        if [ -n "$$IMG" ] ;\
        then \
            docker rmi -f "$$IMG" ;\
        fi ;\
        }


build-process:
	cd $(CODE)/process && \
    docker build --network=host \
    -t $(REGISTRY)/${NAMESPACE}/${PROCESS_IMAGE}:${VERSION} \
    -t $(REGISTRY)/${NAMESPACE}/${PROCESS_IMAGE}:${LATEST} . 

build-train:
	cd $(CODE)/train && \
	docker build --network=host \
	-t $(REGISTRY)/${NAMESPACE}/${TRAIN_IMAGE}:${VERSION} \
	-t $(REGISTRY)/${NAMESPACE}/${TRAIN_IMAGE}:${LATEST} . 


push-process-img:
	echo "Try to push $(REGISTRY)/$(NAMESPACE)/$(PROCESS_IMAGE):$(VERSION) to $(REGISTRY)"
	docker push "$(REGISTRY)/$(NAMESPACE)/$(PROCESS_IMAGE):$(VERSION)"
	echo "Try to push $(REGISTRY)/$(NAMESPACE)/$(PROCESS_IMAGE):$(LATEST) to $(REGISTRY)"
	docker push "$(REGISTRY)/$(NAMESPACE)/$(PROCESS_IMAGE):$(LATEST)"

push-train-img:
	echo "Try to push $(REGISTRY)/$(NAMESPACE)/$(TRAIN_IMAGE):$(VERSION) to $(REGISTRY)"
	docker push "$(REGISTRY)/$(NAMESPACE)/$(TRAIN_IMAGE):$(VERSION)"
	echo "Try to push $(REGISTRY)/$(NAMESPACE)/$(TRAIN_IMAGE):$(LATEST) to $(REGISTRY)"
	docker push "$(REGISTRY)/$(NAMESPACE)/$(TRAIN_IMAGE):$(LATEST)"

