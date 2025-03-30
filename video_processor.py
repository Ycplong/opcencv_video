import os
import cv2
import numpy as np
import time
import pyautogui
import argparse
import threading
import random
import string
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from paddleocr import PaddleOCR
from tqdm import tqdm


class VideoProcessor:
    """视频处理全能工具（录制+关键帧提取+去重+OCR）"""

    def __init__(self):
        # 核心配置参数
        self.config = {
            # 录制配置
            'rec_frame_scale': 0.3,
            'rec_threshold': 0.85,
            'rec_min_interval': 2.0,
            'chunk_duration': 300,

            # 视频处理配置
            'proc_frame_scale': 0.2,
            'base_frame_skip': 5,
            'fast_frame_skip': 30,
            'max_workers': 4,

            # OCR配置
            'ocr_lang': 'ch',
            'ocr_gpu': True
        }
        self._stop_event = threading.Event()
        self.ocr_engine = None

    def _generate_random_filename(self, prefix="rec", extension="mp4"):
        """生成随机文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"{prefix}_{timestamp}_{random_str}.{extension}"

    # ----------------------------
    # 屏幕录制相关方法
    # ----------------------------
    def _preprocess(self, frame, scale):
        """通用预处理"""
        return cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (0, 0),
            fx=scale,
            fy=scale
        )

    def _compare_frames(self, frame1, frame2):
        """优化版帧比较"""
        diff = cv2.absdiff(frame1, frame2)
        return np.sum(diff) / (diff.size * 255)

    def _should_capture(self, current_frame, last_frame, last_capture_time, is_recording=True):
        """智能判断关键帧"""
        if last_frame is None:
            return True

        scale = self.config['rec_frame_scale'] if is_recording else self.config['proc_frame_scale']
        min_interval = self.config['rec_min_interval'] if is_recording else 0

        if (time.time() - last_capture_time) < min_interval:
            return False

        processed_current = self._preprocess(current_frame, scale)
        processed_last = self._preprocess(last_frame, scale)
        similarity = self._compare_frames(processed_last, processed_current)

        threshold = self.config['rec_threshold'] if is_recording else self.config.get('proc_threshold', 0.9)
        return similarity < threshold

    def _record_chunk(self, output_dir, duration, fps):
        """录制分块视频"""
        screen_size = pyautogui.size()
        video_path = os.path.join(output_dir, self._generate_random_filename())
        os.makedirs(output_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, screen_size)

        start_time = time.time()
        last_frame = None
        last_capture = 0

        while not self._stop_event.is_set() and (time.time() - start_time < duration):
            try:
                img = pyautogui.screenshot()
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                out.write(frame)

                if self._should_capture(frame, last_frame, last_capture, is_recording=True):
                    self._save_keyframe(frame, os.path.join(output_dir, "keyframes"))
                    last_capture = time.time()
                    last_frame = frame.copy()

                time.sleep(max(0, 1 / fps - (time.time() % (1 / fps))))
            except Exception as e:
                print(f"⚠️ 捕获异常: {str(e)}")
                continue

        out.release()
        return video_path

    def _save_keyframe(self, frame, output_dir):
        """保存关键帧"""
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, self._generate_random_filename(prefix="keyframe", extension="jpg"))
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    def start_recording(self, output_dir="recordings", duration=3600, fps=15):
        """启动长时间录制"""
        print(f"🎥 开始录制 {duration}秒 (Ctrl+C停止)...")
        print(f"📂 视频将保存至: {os.path.abspath(output_dir)}")

        try:
            with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
                futures = []
                start_time = time.time()

                while not self._stop_event.is_set() and (time.time() - start_time < duration):
                    future = executor.submit(
                        self._record_chunk,
                        output_dir,
                        min(self.config['chunk_duration'], duration - (time.time() - start_time)),
                        fps
                    )
                    futures.append(future)

                for future in futures:
                    print(f"✅ 视频段已保存: {future.result()}")
        except KeyboardInterrupt:
            print("\n🛑 用户停止录制")
        finally:
            self._stop_event.set()
            print(f"📁 关键帧保存在: {os.path.join(output_dir, 'keyframes')}")

    # ----------------------------
    # 视频处理相关方法
    # ----------------------------
    def process_video(self, video_path, output_dir, threshold=0.9, fast_mode=False):
        """快速处理视频提取关键帧"""
        start_time = datetime.now()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件 {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_skip = self.config['fast_frame_skip'] if fast_mode else self.config['base_frame_skip']
        scale = self.config['proc_frame_scale'] if fast_mode else 0.5

        print(f"\n🔍 开始处理: {os.path.basename(video_path)}")
        print(f"📊 总帧数: {total_frames} | 模式: {'极速' if fast_mode else '标准'}")
        print(f"⚙️ 参数: 缩放={scale}x 跳帧={frame_skip} 阈值={threshold}")

        os.makedirs(output_dir, exist_ok=True)
        keyframe_count = 0
        last_keyframe = None

        def process_frame(pos, frame):
            nonlocal last_keyframe, keyframe_count
            if last_keyframe is None or self._should_capture(frame, last_keyframe, 0, is_recording=False):
                filename = os.path.join(output_dir, self._generate_random_filename(prefix="frame", extension="jpg"))
                cv2.imwrite(filename, frame)
                last_keyframe = frame.copy()
                keyframe_count += 1
                return True
            return False

        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            futures = []
            pos = 0

            while True:
                ret, frame = cap.read()
                if not ret: break

                if pos % frame_skip == 0:
                    futures.append(executor.submit(process_frame, pos, frame))

                pos += 1
                if pos % 100 == 0:
                    print(f"\r🚀 已处理 {pos}/{total_frames} 帧 | 关键帧: {keyframe_count}", end="")

            for future in futures:
                future.result()

        cap.release()
        print(f"\n✅ 处理完成！共提取 {keyframe_count} 关键帧")
        print(f"⏱️ 耗时: {(datetime.now() - start_time).total_seconds():.1f}秒")

    # ----------------------------
    # 帧去重相关方法
    # ----------------------------
    def deduplicate_frames(self, frame_folder, threshold=0.95, preview=False):
        """
        快速视频帧去重
        :param frame_folder: 帧图片文件夹路径
        :param threshold: 相似度阈值(0-1)
        :param preview: 是否显示处理中的帧对比
        :return: 需要删除的文件列表
        """
        frames = sorted([f for f in os.listdir(frame_folder)
                         if f.lower().endswith(('png', 'jpg', 'jpeg'))])

        if not frames:
            print("未找到任何图片文件！")
            return []

        to_delete = []
        prev_gray = None
        total_frames = len(frames)
        start_time = time.time()

        print(f"开始处理 {total_frames} 帧...")
        print("进度: 0%", end="", flush=True)

        for i, frame_file in enumerate(frames):
            current_path = os.path.join(frame_folder, frame_file)
            current_gray = cv2.imread(current_path, cv2.IMREAD_GRAYSCALE)

            if current_gray is None:
                print(f"\n警告: 无法读取文件 {frame_file}，跳过")
                continue

            if prev_gray is not None:
                small_prev = cv2.resize(prev_gray, (64, 64))
                small_current = cv2.resize(current_gray, (64, 64))

                similarity = ssim(small_prev, small_current,
                                  win_size=3,
                                  data_range=small_current.max() - small_current.min())

                if similarity > threshold:
                    to_delete.append(frame_file)

                    if preview:
                        compare_img = np.hstack((prev_gray, current_gray))
                        cv2.putText(compare_img, f"Similarity: {similarity:.3f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.imshow('Frame Comparison', compare_img)
                        cv2.waitKey(1)
                else:
                    prev_gray = current_gray
            else:
                prev_gray = current_gray

            progress = (i + 1) / total_frames * 100
            print(
                f"\r进度: {progress:.1f}% | 已处理: {i + 1}/{total_frames} | 重复: {len(to_delete)} | 耗时: {time.time() - start_time:.1f}s",
                end="", flush=True)

        if preview:
            cv2.destroyAllWindows()

        print(f"\n处理完成！总耗时: {time.time() - start_time:.2f}秒")
        return to_delete

    # ----------------------------
    # OCR文字提取相关方法
    # ----------------------------
    def init_ocr(self):
        """初始化OCR引擎"""
        if self.ocr_engine is None:
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang=self.config['ocr_lang'],
                use_gpu=self.config['ocr_gpu']
            )

    def extract_text_from_images(self, folder_path, output_txt='output.txt'):
        """提取文件夹中所有图片的文字"""
        self.init_ocr()
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = [f for f in os.listdir(folder_path)
                       if os.path.splitext(f)[1].lower() in valid_extensions]

        if not image_files:
            print("文件夹中没有找到支持的图片文件")
            return

        image_files.sort()

        with open(output_txt, 'w', encoding='utf-8') as f_out:
            for img_file in tqdm(image_files, desc="处理图片中"):
                img_path = os.path.join(folder_path, img_file)

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"无法读取图片: {img_file}")
                        continue

                    result = self.ocr_engine.ocr(img, cls=True)

                    if result and result[0]:
                        f_out.write(f"\n\n----- {img_file} -----\n")
                        for line in result[0]:
                            if line and line[1]:
                                text = line[1][0]
                                f_out.write(text + "\n")
                except Exception as e:
                    print(f"处理图片 {img_file} 时出错: {str(e)}")

        print(f"所有图片文字已提取到: {output_txt}")


def main():
    parser = argparse.ArgumentParser(description="视频处理全能工具")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # 录制命令
    rec_parser = subparsers.add_parser('record', help='屏幕录制')
    rec_parser.add_argument("--output", default="recordings", help="输出目录")
    rec_parser.add_argument("--duration", type=int, default=3600, help="录制时长(秒)")
    rec_parser.add_argument("--fps", type=int, default=15, help="帧率")

    # 处理命令
    proc_parser = subparsers.add_parser('process', help='处理视频')
    proc_parser.add_argument("-f", required=True, help="视频路径")
    proc_parser.add_argument("-d", default="./output", help="输出目录")
    proc_parser.add_argument("--fast", action="store_true", help="极速模式")
    proc_parser.add_argument("--threshold", type=float, default=0.9, help="敏感度阈值")

    # 去重命令
    dedup_parser = subparsers.add_parser('dedup', help='帧去重')
    dedup_parser.add_argument("-i", required=True, help="帧图片目录")
    dedup_parser.add_argument("-t", type=float, default=0.95, help="相似度阈值")
    dedup_parser.add_argument("--preview", action="store_true", help="预览模式")

    # OCR命令
    ocr_parser = subparsers.add_parser('ocr', help='文字提取')
    ocr_parser.add_argument("-i", required=True, help="图片目录")
    ocr_parser.add_argument("-o", default="output.txt", help="输出文本文件")

    args = parser.parse_args()
    processor = VideoProcessor()

    if args.command == 'record':
        processor.start_recording(
            output_dir=args.output,
            duration=args.duration,
            fps=args.fps
        )
    elif args.command == 'process':
        processor.process_video(
            video_path=args.f,
            output_dir=args.d,
            threshold=args.threshold,
            fast_mode=args.fast
        )
    elif args.command == 'dedup':
        dup_frames = processor.deduplicate_frames(
            frame_folder=args.i,
            threshold=args.t,
            preview=args.preview
        )
        print(f"\n发现 {len(dup_frames)} 个重复帧")
        if dup_frames:
            print("示例重复帧:", dup_frames[:10])
            confirm = input("确认删除这些文件吗？(y/n): ")
            if confirm.lower() == 'y':
                for f in dup_frames:
                    try:
                        os.remove(os.path.join(args.i, f))
                    except Exception as e:
                        print(f"删除 {f} 失败: {e}")
                print("删除完成！")
    elif args.command == 'ocr':
        processor.extract_text_from_images(
            folder_path=args.i,
            output_txt=args.o
        )


if __name__ == "__main__":
    main()