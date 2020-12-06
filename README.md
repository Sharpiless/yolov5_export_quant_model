# yolov5模型训练后量化代码

在终端运行：

```bash
python slim.py --in_weights last.pt --out_weights slim_model.pt --device 0
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201206114606779.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)
可以看到权重文件压缩到了 43 MB。
