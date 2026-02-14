# syntax=docker/dockerfile:1.9

FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime AS devbox

ARG UID=1001
ARG GID=1001
ARG USERNAME=dev
ENV HOME="/home/${USERNAME}"

RUN apt-get update -q \
    && apt-get install -y --no-install-recommends \
    # Generic devbox stuff
    curl sudo tmux less rsync git git-lfs vim ssh telnet make \
    # Fancy devbox stuff
    ripgrep docker-buildx \
    # C++ compilation tools
    cmake ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and grant password-less sudo permissions
RUN addgroup --gid $GID ${USERNAME} \
    && adduser --uid $UID --gid $GID --disabled-password --gecos "" ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
    && chown "${USERNAME}:${USERNAME}" /opt \
    && chown "${USERNAME}:${USERNAME}" /workspace

# Install Kubectl
ARG KUBECTL_VERSION=v1.32.8
RUN curl -LO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/

# Install Backblaze B2 CLI
RUN curl -LO "https://github.com/Backblaze/B2_Command_Line_Tool/releases/latest/download/b2-linux" \
    && chmod +x b2-linux \
    && mv b2-linux /usr/local/bin/b2


# Install Claude Code and Gemini CLI. First install node and npm, then install the assistants.
RUN curl -sL https://deb.nodesource.com/setup_20.x | bash - \
    && apt install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# We need to chown $HOME because the npm install -g command somehow creates root-owned files there.
RUN npm install -g @google/gemini-cli @anthropic-ai/claude-code \
    && sudo chown $UID:$GID -R "$HOME"

# Install uv in /opt/local/bin
USER ${UID}
ENV XDG_BIN_HOME=/opt/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

#####################
# UV settings: we ensure that UV and associated python versions are *not* installed under /home/dev, because that would be
# 'overwritten' when mounting the persistent user storage to /home/dev.
#####################

# `UV_COMPILE_BYTECODE` influences both `uv sync`, `uv pip` commands AND `uv run` commands. It introduces a 5-6 second wait for
# the latter. So we disable it by default, and instead enable it individually for every installation command.
ENV UV_COMPILE_BYTECODE=0
ENV UV_LINK_MODE=hardlink

# Put cache (including uv) in /opt/cache and /opt/cache/uv
ENV XDG_CACHE_HOME="/opt/cache"
ENV UV_CACHE_DIR="${XDG_CACHE_HOME}/uv"

# Put all other uv storage in /opt/uv
ENV UV_ROOT_DIR="/opt/uv"
ENV UV_PYTHON_INSTALL_DIR="${UV_ROOT_DIR}/python"
ENV UV_PROJECT_ENVIRONMENT="${UV_ROOT_DIR}/venv"
ENV UV_TOOL_DIR="${UV_ROOT_DIR}/tools"

# Set Path to include UV and venv bin directories so we don't have to use `uv run` every time.
ENV PATH="${XDG_BIN_HOME}:${UV_PROJECT_ENVIRONMENT}/bin:/usr/local/cuda/bin:${HOME}/.local/bin:$PATH"

# Most of the time this will get overwritten if we mount a volume to /home/dev. But if it's not we still want to display the
# message.
RUN echo "cat /etc/motd" >> "${HOME}/.bashrc"
ENTRYPOINT ["/bin/bash"]

# Set the non-root user as the default user
USER ${USERNAME}
WORKDIR /home/${USERNAME}
