l6.py文件内包含了以transformer为分割头的整个模型，在代码的最终阶段有train和predict的逻辑，因此如果想训练则直接不注释任何代码run即可；
如果是要加载模型权重进行预测则将train_level3()注释即可(目前处于注释状态)；(若要测试直接run即可)
本次上传的代码包含了将心室和息肉混合而成的数据集，但该模型也支持单独对心室或息肉的数据集进行训练，需使用者自行使用单独的数据集并依据dataload函数的逻辑更改相应的路径即可
