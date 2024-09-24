**本项目为医学影像运动伪影识别**  

**组员**：  
>汪浩 20211120272  
谷道斌 20211120274  
伍秋风 20211050211  
张秋武 20211050212  
袁震东 20211120025  

**代码解释**：  
>data文件夹为数据文件夹
>
>LSTMClassifier文件夹为模型权重和训练图像文件夹
>
logger文件夹为模型训练的日志文件 

dataset.py为数据的处理方法 

model.py为模型代码文件  

train.py为模型训练文件，其中包含测试与验证方法  

utils.py为工具代码，其中包含绘图，初始化日志，创建文件夹等  

test.py为测试的方法  

[注]测试时，测试数据的文件应该为一个.csv文件，其中.csv应有两列  
如下：  
data/images/00050.dcm,good  
前面为图像路径，后面为标签  

**测试**  
```shell
python test.py --test_file_path "./data/test_val_set.csv"
```


    
