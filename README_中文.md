# 大众汽车品牌特征分析代码

版本0.1.0。包含三个独立模块，保留本次研究使用的主要参数：

1. `vgg_gradcam.py`：VGG-16＋双层Grad-CAM，前景显示掩码、Gamma=0.65。
2. `euclidean_distance.py`：朗逸LA、帕萨特PA、速腾SA的九类部件距离，输出Table2、Table3，按rank(μ)升序。
3. `gm11_forecast.py`：11_24标准GM(1,1)，用于LG、CL、TL五期轮廓的下一阶段外推。

专家讨论、Bézier描绘、BGF/BDF最终分类属于人工环节；代码不自动替代这些环节。专家小组人数按论文统一为15人。

## 安装和运行

使用Python 3.12，在项目根目录运行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe vgg_gradcam.py --input data/vehicles --output outputs/gradcam
.\.venv\Scripts\python.exe euclidean_distance.py --input data/components --output outputs/euclidean
.\.venv\Scripts\python.exe gm11_forecast.py --input data/sa_history --output outputs/gm11
```

第一次运行VGG时由torchvision下载官方ImageNet V1权重；不必上传约528MiB的权重文件。
如本机已有缓存，VGG命令后加`--torch-home 缓存目录`即可。
这份环境版本来自实际核查环境，其他Python或依赖版本未全部验证。

## 输入规范

**整车：**放入`data/vehicles`，采用I1_Lavida_front.png、I2_Lavida_back.png、I3_Passat_front.png、I4_Passat_back.png、I5_Sagitar_front.png、I6_Sagitar_back.png。
核查图像为448×448、纯灰背景。车窗和车牌等预处理在导入前完成，代码不自动寻找这些区域，车标保留。
非方图等比例填充为448×448，模型输入224×224；灰度三通道和ImageNet标准化。

**部件：**放入`data/components`，命名`部件_车型.png`。车型LA、PA、SA；部件FL、CL、UG、LG、BD、RT、TL、RRR、RVM，共27张。
核查图像为340×340、不透明白底，保留原有颜色、位置与画布占比。BO、MA不进入默认计算。
RGB图像送入最后池化层，512×7×7展平为25,088维，不做L2归一化；SD分母n−1，S权重0.7/0.3。

**大众历史轮廓：**放入`data/sa_history`，每类LG、CL、TL各五张，命名如`CL_SA_2015.png`、`CL_SA_2018.png`、`CL_SA_2020.png`、`CL_SA_2023.png`、`CL_SA_2026.png`。
采用11_24的红色轮廓/红色填充白底输入；同部件各期画布、单位、左右侧、半幅/全幅和描绘范围应一致。
原来每次一组的`curve1.png`至`curve5.png`也支持：命令后加`--legacy-part CL`，按对应部件选择LG/CL/TL。

## 灰色预测的三个区分

- 默认`--solver stable`：数值稳定实现，保留同一标准GM模型，处理近零参数、常数序列；这是明确披露的数值调整，不是残差修正。
- `--solver legacy`：保留原11_24矩阵求逆与累计响应相减公式用于追溯；常数或近常数坐标可能出现除零或异常预测。
- 显示参数`--smoothing periodic`为原版周期三次插值；`--smoothing light`为后来讨论的轻度显示平滑；`none`为折线。

所有模式均保存原始100点预测坐标。显示平滑坐标单独标记`DISPLAY_ONLY`，不可直接替换原始坐标进行误差评价。
原始OpenCV顺序不保证跨阶段语义点对应；本命令不自动旋转、镜像、对齐起点或重新缩放历史曲线。
输出的下一阶段不自动命名为某个未来年份；它不是留出2026年的验证实验。

## 本次整理范围

代码、三份无输出笔记本、依赖、输入规范、原函数追溯、数值回归检查和合成示例生成器均可公开上传。
不包含论文Word、问卷个人数据、网上原始照片、模型权重、本地虚拟环境或旧输出目录。
已找到的六张大众整车图及27张部件图用于本地核查；大众五期原始历史轮廓未找到完整独立文件，因此不声称已复现论文历史预测图。
完整核查情况见`docs/verification.md`；文件哈希见`docs/located_input_manifest.json`。

## 使用笔记本

`notebooks`内三个ipynb调用同目录项目脚本。Jupyter从项目根目录启动即可；Colab需先上传并解压整个项目，不能只上传单个ipynb。
需要修改输入目录时，在笔记本第二个单元修改command中的路径。

## 公开上传

按照`docs/GITHUB_UPLOAD_中文.md`操作。将解压后的项目内容上传到GitHub仓库根目录，不要只上传zip。
发布前由作者自行选择许可证；目前没有替作者授予额外的代码再使用许可。
