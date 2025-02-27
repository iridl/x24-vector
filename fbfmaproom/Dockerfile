FROM centos:7.9.2009 as common

ARG PIXI_VERSION="v0.41.4"

COPY docker/CentOS-Vault.repo /etc/yum.repos.d/
RUN rm /etc/yum.repos.d/CentOS-Base.repo

RUN yum install -y httpd

RUN rm -r /etc/httpd/conf.d/* /etc/httpd/conf.modules.d



FROM common as build

# httpd-devel and gcc are to support building mod_wsgi from source.
RUN yum install -y httpd-devel gcc

# install pixi
RUN curl --silent --location \
    https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz \
    | tar zxf - --to-stdout \
    > /usr/local/bin/pixi \
    && chmod +x /usr/local/bin/pixi

# build pixi environment
WORKDIR /app
COPY pixi.lock pixi.toml .
RUN pixi install --locked --environment prod
RUN pixi shell-hook --locked --shell bash --environment prod > activate

# mod_wsgi: use pip to compile mod_wsgi from source for the particular versions
# of apache and python that we're using.
RUN pixi run --locked --environment prod pip install mod_wsgi==4.7.1



FROM common

COPY --from=build /app /app
COPY . /app

# httpd config
COPY docker/httpd.conf /etc/httpd/conf/httpd.conf

# The following is bad security practice if running httpd as
# root, but we will run it as apache.
RUN chmod g+rwx /run/httpd

USER apache:apache
WORKDIR /app
ENTRYPOINT ["/app/docker/entrypoint"]
CMD ["/app/docker/service"]
