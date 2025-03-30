# 🎥 智能视频处理与OCR工具

## 📋 功能概览
- **视频录制**：智能截取关键帧，自动分段存储
- **视频处理**：高效提取关键帧，支持极速模式
- **图像去重**：基于SSIM算法精准识别相似帧
- **OCR识别**：多语言文字提取，支持GPU加速

## 🛠️ 快速开始

### 安装依赖
```bash
pip install opencv-python numpy pyautogui scikit-image paddleocr tqdm
```

### 基本使用

#### 1. 屏幕录制
```bash
python video_processor.py record \
    --output my_recordings \
    --duration 1800 \
    --fps 12
```

#### 2. 视频处理
```bash
python video_processor.py process \
    -f input.mp4 \
    -d output_frames \
    --fast
```

#### 3. 图像去重
```bash
python video_processor.py dedup \
    -i frames_folder \
    -t 0.92
```

#### 4. 文字提取
```bash
python video_processor.py ocr \
    -i images_folder \
    -o result.txt
```

## ⚙️ 参数详解

### 录制模式
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output` | 输出目录 | recordings |
| `--duration` | 录制时长(秒) | 3600 |
| `--fps` | 帧率 | 15 |
| `--chunk` | 分片时长(秒) | 300 |

### 处理模式
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-f` | 输入视频路径 | 必填 |
| `-d` | 输出目录 | output |
| `--fast` | 启用极速模式 | False |
| `--threshold` | 敏感度阈值 | 0.9 |

### 去重模式
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-i` | 图片目录 | 必填 |
| `-t` | 相似度阈值 | 0.95 |
| `--preview` | 启用预览 | False |

### OCR模式
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-i` | 图片目录 | 必填 |
| `-o` | 输出文件 | output.txt |
| `--lang` | 语言(ch/en/ch_en) | ch |

## 💡 使用技巧
1. 录制PPT演示时建议设置`--threshold 0.92`
2. 处理动态内容时可启用`--fast`模式
3. OCR识别前建议先进行图像去重
4. GPU环境下请安装paddlepaddle-gpu提升性能

## 🐛 常见问题
**Q: 如何提升OCR识别准确率？**
A: 确保图像清晰度，可尝试：
- 提高原始图像分辨率
- 增加图像对比度
- 使用`--lang ch_en`中英混合模式

**Q: 去重阈值如何选择？**
A: 推荐值：
- 严格去重：0.93-0.95
- 一般去重：0.90-0.92
- 宽松去重：0.85-0.89

**Q: 为什么极速模式处理更快？**
A: 极速模式采用：
- 30帧跳帧技术
- 0.2倍图像缩放
- 多线程并行处理