# 新建公开仓库并上传

建议仓库名称：`vehicle-brand-feature-volkswagen`

建议简介：`VGG-16/Grad-CAM, component similarity and GM(1,1) curve forecasting for Volkswagen vehicle form analysis.`

1. 登录GitHub，打开 https://github.com/new 。Owner选择你的账号，填写上述仓库名称。
2. Visibility选择Public。为便于在网页找到上传入口，可勾选Add a README file。先不选择额外gitignore模板或许可证，项目已带.gitignore，许可证由作者决定。
3. 点击Create repository。
4. 解压本地代码包，打开其中的`vehicle-brand-feature-volkswagen`文件夹。
5. 在GitHub仓库选择Add file → Upload files，将该文件夹**里面的文件及子文件夹**拖到上传区域，保持README.md与三个py脚本在仓库根目录。
6. 提交说明填写`Add Volkswagen feature analysis and forecasting code`，点击Commit changes。已有的README.md会由项目说明替换。
7. 检查仓库根目录是否直接显示README.md、vgg_gradcam.py、euclidean_distance.py、gm11_forecast.py及notebooks文件夹。
8. 在仓库的Releases创建`v0.1.0`标签和版本，或记录本次提交的完整commit ID，用于论文中的代码版本引用。

上传的是源代码及说明，不是整个本地工作目录。虚拟环境、.torch-cache、outputs、权重、本地原始研究图片不在整理包内。
网页上传单个文件上限25MiB，本包文件均低于此上限。GitHub网页上传也应逐项检查文件，不能只依赖.gitignore排除误拖入的文件。

官方说明：
- https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
- https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository

论文代码可用性说明可在实际发布后填写：

The implementation code, dependency specifications, parameter settings, and reproduction instructions are available at [actual repository URL], version v0.1.0.

目前没有代填仓库地址，也未声称已在GitHub发布。代码公开不等于原始数据已经公开；数据可用性须另据实际情况说明。
