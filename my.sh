docker run --rm -ti -v "$(pwd)":/data --network=host debian:10.13 bash
docker run --rm -ti -v "$(pwd)":/data --network=host docker.art.haizhi.com/dmc/calradia-x86_64 bash



sed -i "s@http://deb.debian.org@http://mirrors.cloud.tencent.com@g" /etc/apt/sources.list \
    && rm -Rf /var/lib/apt/lists/* \
    && apt-get update

apt install -y wget curl unzip

curl https://install.duckdb.org | sh

wget 'https://art.haizhi.com:443/artifactory/bower/duckdb/duckdb_cli-linux-amd64.v1_3_1.zip'
unzip duckdb_cli-linux-amd64.v1_3_1.zip
chmod +x duckdb
mv duckdb /usr/local/bin/

duckdb
INSTALL vss;
INSTALL fts;


# sqlite

cd ~/data1/hipporag
docker run --name zts-hipporag -dti -v "$(pwd)":/data --network=host docker.art.haizhi.com/ai/alita-x86_64:ai_1.9.5 bash
docker exec -ti zts-hipporag bash

export https_proxy=http://192.168.1.167:7890 http_proxy=http://192.168.1.167:7890 all_proxy=socks5://192.168.1.167:7890

# build libsimple
apt update
apt install -y gcc make libreadline-dev build-essential libssl-dev git wget curl
wget https://github.com/Kitware/CMake/releases/download/v3.19.8/cmake-3.19.8.tar.gz
tar -zvxf cmake-3.19.8.tar.gz
cd cmake-3.19.8
./bootstrap
make -j 8
make install

git clone https://github.com/wangfenjin/simple
cd simple
git submodule init
git submodule update
mkdir build; cd build
cmake .. -DBUILD_TEST_EXAMPLE=OFF
make -j 8
make install


pip install sqlite-vss
apt install -y libblas3 liblapack3

# https://github.com/wangfenjin/simple

docker pull debian:12.11

> .load /data/libsimple-ubuntu-2204-x86_64/libsimple

# python lib
pip install sqlite-vss python_igraph==0.11.8

# 容器启动ssh
```shell
apt install openssh-server -y
vi /etc/ssh/sshd_config
mkdir -p /run/sshd
chmod 755 /run/sshd
/usr/sbin/sshd
```

```
# 启用密码认证
PasswordAuthentication yes

# 如果使用 root 登录，也需要设置：
PermitRootLogin yes      # 可选（不建议生产环境开启）

# 禁用 GSSAPI 认证（否则登录会卡顿）
GSSAPIAuthentication no

Port 38129
```
