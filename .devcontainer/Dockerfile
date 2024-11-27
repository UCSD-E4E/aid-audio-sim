FROM ubuntu:24.04

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    sudo \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libglm-dev \
    libegl1-mesa-dev \
    ninja-build \
    xorg-dev \
    freeglut3-dev \
    pkg-config \
    wget \
    zip \
    lcov\
    libhdf5-dev \
    libomp-dev \
    unzip \
    make \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo 'ubuntu ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

USER ubuntu
ENV HOME="/home/ubuntu"
RUN mkdir -p ${HOME}
WORKDIR ${HOME}

RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN echo "export PYENV_ROOT=\"\$HOME/.pyenv\"" >> ${HOME}/.bashrc && \
	echo "[[ -d \$PYENV_ROOT/bin ]] && export PATH="\$PYENV_ROOT/bin:\$PATH"" >> ${HOME}/.bashrc && \
	echo "eval \"\$(pyenv init -)\"" >> ${HOME}/.bashrc && \
	echo "\"$(pyenv virtualenv-init -)\"" >> ${HOME}/.bashrc

RUN pyenv install 3.9
RUN pyenv global 3.9

RUN git clone --recursive https://github.com/facebookresearch/habitat-sim.git && \
    cd ${HOME}/habitat-sim && \
    git checkout v0.3.2 && \
    pip install --upgrade pip && pip install -r requirements.txt && pip cache purge && \
    python setup.py install --audio --headless && \
    cd ${HOME} && \
    rm -rf habitat-sim