FROM centos:7.9.2009

# yum packages
# httpd-devel and gcc are to support building mod_wsgi from source.
RUN yum install -y httpd httpd-devel gcc

RUN rm -r /etc/httpd/conf.d /etc/httpd/conf.modules.d


# miniconda
RUN curl -L \
    https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh \
    -o /miniconda-installer.sh && \
    bash /miniconda-installer.sh -b -p /conda && \
    eval "$('/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" && \
    conda config --set auto_update_conda False && \
    conda install conda==4.10.1

# conda environment
COPY environment_linux.yml /build/environment.yml
RUN eval "$('/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" && \
    conda env create -f /build/environment.yml -n app

# mod_wsgi: use pip to compile mod_wsgi from source for the particular versions
# of apache and python that we're using.
RUN eval "$('/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" && \
    conda activate app && \
    pip install mod_wsgi==4.7.1

RUN yum -y install mesa-libGL

# httpd config
COPY docker/httpd.conf /etc/httpd/conf/httpd.conf
# The following is bad security practice if running httpd as
# root, but we will run it as apache.
RUN chmod g+rwx /run/httpd

# install application
COPY . /app/

ENV CONFIG=/app/config-sample.yaml

USER apache:apache
WORKDIR /app
ENTRYPOINT ["/app/docker/entrypoint"]
CMD ["/app/docker/service"]
