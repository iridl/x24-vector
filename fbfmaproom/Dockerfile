ARG BASE=centos:7.9.2009
FROM ${BASE} as build

# yum packages
# httpd-devel and gcc are to support building mod_wsgi from source.
# mesa-libGL is a dependency of opencv that is not packaged for conda (see
# https://conda-forge.org/docs/maintainer/knowledge_base.html#core-dependency-tree-packages-cdts)
RUN yum install -y httpd httpd-devel gcc mesa-libGL

RUN rm -r /etc/httpd/conf.d /etc/httpd/conf.modules.d


# miniconda
RUN curl -L \
    https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh \
    -o /miniconda-installer.sh
RUN bash /miniconda-installer.sh -b -p /conda
RUN eval "$('/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" && \
    conda config --set auto_update_conda False && \
    conda install conda==4.11.0

# build conda environment
COPY fbfmaproom/environment_linux.yml /build/environment.yml
RUN eval "$('/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" && \
    conda env create -f /build/environment.yml && \
    conda clean -a

# mod_wsgi: use pip to compile mod_wsgi from source for the particular versions
# of apache and python that we're using.
RUN eval "$('/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" && \
    conda activate fbfmaproom2 && \
    pip install mod_wsgi==4.7.1



FROM ${BASE}

COPY --from=build /conda /conda

RUN yum install -y httpd mesa-libGL

# httpd config
COPY fbfmaproom/docker/httpd.conf /etc/httpd/conf/httpd.conf
# The following is bad security practice if running httpd as
# root, but we will run it as apache.
RUN chmod g+rwx /run/httpd

# install application
COPY fbfmaproom/ /app/
COPY pingrid.py /app/pingrid.py

USER apache:apache
WORKDIR /app
ENTRYPOINT ["/app/docker/entrypoint"]
CMD ["/app/docker/service"]
