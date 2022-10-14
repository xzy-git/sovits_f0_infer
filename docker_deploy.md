# 基于docker GPU部署

构建docker镜像

```bash
bash ./build_docker.sh
```

下载模型文件，模型文件存储在任意地方，随后修改[docker-compose.yml](docker-compose.yml)里的模型挂载目录

使用docker-compose启动服务

```bash
docker-compose up
```