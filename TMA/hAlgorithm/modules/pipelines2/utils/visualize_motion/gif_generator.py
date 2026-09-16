import glob
import logging
import os
import re

logger = logging.getLogger(__name__)


class GifGenerator:
    """Generate animated GIF from saved flow visualisation images."""

    @staticmethod
    def _get_pil_font(size=18):
        from PIL import ImageFont

        fonts = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        for f in fonts:
            if os.path.exists(f):
                return ImageFont.truetype(f, size)
        return ImageFont.load_default()

    @staticmethod
    def _add_pil_label(img, label, position="top", bg_color=(0, 0, 0, 180), text_color=(255, 255, 255)):
        from PIL import Image, ImageDraw

        img = img.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        font = GifGenerator._get_pil_font(16)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        x = (img.width - text_w) // 2
        y = 4 if position == "top" else (img.height - text_h - 8 if position == "bottom" else 4)

        draw.rectangle([x - 4, y - 4, x + text_w + 4, y + text_h + 4], fill=bg_color)
        draw.text((x, y), label, fill=text_color, font=font)
        return Image.alpha_composite(img, overlay).convert("RGB")

    @staticmethod
    def generate_flow_gif(flow_img_dir, output_path=None, fps=2.0, size=280):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            logger.debug("[VisMotion] PIL not found, skip gif generation")
            return None

        output_path = output_path or os.path.join(flow_img_dir, "flow_visualization.gif")

        def _get_map(pattern, files):
            return {
                tuple(int(x) for x in re.search(pattern, f).groups()): f
                for f in files
                if re.search(pattern, f)
            }

        rgb_map = _get_map(r"rgb_f(\d+)", glob.glob(os.path.join(flow_img_dir, "rgb_f*.png")))
        mag_map = _get_map(
            r"magnitude_f(\d+)_to_f(\d+)",
            glob.glob(os.path.join(flow_img_dir, "magnitude_f*.png")),
        )
        xyz_map = _get_map(
            r"flow_xyz_f(\d+)_to_f(\d+)",
            glob.glob(os.path.join(flow_img_dir, "flow_xyz_f*.png")),
        )

        frame_pairs = sorted(set(mag_map.keys()) | set(xyz_map.keys()))
        if not frame_pairs:
            return None

        def load_resize(path):
            if path and os.path.exists(path):
                return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
            img = Image.new("RGB", (size, size), (50, 50, 50))
            ImageDraw.Draw(img).text((size // 4, size // 2), "N/A", fill=(128, 128, 128))
            return img

        frames = []
        for ref_idx, tgt_idx in frame_pairs:
            rgb_ref = GifGenerator._add_pil_label(
                load_resize(rgb_map.get((ref_idx,))), f"Reference (F{ref_idx})"
            )
            rgb_tgt = GifGenerator._add_pil_label(
                load_resize(rgb_map.get((tgt_idx,))), f"Target (F{tgt_idx})"
            )
            mag = GifGenerator._add_pil_label(
                load_resize(mag_map.get((ref_idx, tgt_idx))),
                f"Magnitude (F{ref_idx}->F{tgt_idx})",
            )
            xyz = GifGenerator._add_pil_label(
                load_resize(xyz_map.get((ref_idx, tgt_idx))),
                f"Flow XYZ (F{ref_idx}->F{tgt_idx})",
            )

            comp = Image.new("RGB", (size * 2 + 4, size * 2 + 28), (30, 30, 30))
            comp.paste(rgb_ref, (0, 0))
            comp.paste(rgb_tgt, (size + 4, 0))
            comp.paste(mag, (0, size + 4))
            comp.paste(xyz, (size + 4, size + 4))

            draw = ImageDraw.Draw(comp)
            txt = f"Frame: {ref_idx} -> {tgt_idx}"
            font = GifGenerator._get_pil_font(14)
            tw = draw.textbbox((0, 0), txt, font=font)[2]
            x, y = (comp.width - tw) // 2, comp.height - 20
            draw.rectangle([x - 6, y - 2, x + tw + 6, y + 18], fill=(0, 0, 0))
            draw.text((x, y), txt, fill=(255, 255, 0), font=font)
            frames.append(comp)

        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / fps),
                loop=0,
            )
            logger.info("[GIF] Saved: %s (%d frames)", output_path, len(frames))
        return output_path
