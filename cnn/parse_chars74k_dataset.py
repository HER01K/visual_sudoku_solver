import struct
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps

def load_chars74k(path):
    path = Path(path)
    examples = []

    for digit in range(1, 10):
        folder = path / f"Sample{digit + 1:03d}"
        examples += [(p, digit) for p in sorted(folder.glob("*.png"))]

    return examples


def normalize_img(img, threshold=128, box=20, size=28):
    arr = np.asarray(img)
    mask = arr > threshold
    if not mask.any():
        return np.zeros((size, size), dtype=np.uint8)

    # Find where digit is in image
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))

    # Crop image to just digit
    crop = arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].astype(np.uint8)
    crop_h, crop_w = crop.shape

    scale = box / max(crop_h, crop_w)
    w = max(1, round(crop_w * scale))
    h = max(1, round(crop_h * scale))
    crop = Image.fromarray(crop).resize((w, h), Image.Resampling.LANCZOS)

    # Center crop in canvas using center of mass
    c = np.asarray(crop, dtype=np.float64)
    total = c.sum() or 1.0
    ys, xs = np.indices(c.shape)

    center_x = (xs * c).sum() / total
    center_y = (ys * c).sum() / total
    center = (size - 1) / 2

    offset_x = int(np.clip(round(center - center_x), 0, size - w))
    offset_y = int(np.clip(round(center - center_y), 0, size - h))

    canvas = Image.new("L", (size, size), 0)
    canvas.paste(crop, (offset_x, offset_y))

    # Normalize values so that brightest value in image is set to 255
    out = np.asarray(canvas, dtype=np.float32)
    peak = out.max()
    if peak > 0:
        out *= 255.0 / peak

    return out.round().astype(np.uint8)


def augment_img(img, rng):
    img = ImageOps.expand(img, border=32, fill=0)

    # width stretch
    w, h = img.size
    stretch = rng.uniform(0.8, 1.2)
    img = img.resize((max(1, round(w * stretch)), h), Image.Resampling.BILINEAR)

    # shear
    w, h = img.size
    shear = rng.uniform(-0.2, 0.2)
    img = img.transform((w, h), Image.Transform.AFFINE, (1, shear, -shear * h / 2, 0, 1, 0), resample=Image.Resampling.BILINEAR, fillcolor=0)

    # rotate
    img = img.rotate(rng.uniform(-9, 9), resample=Image.Resampling.BILINEAR, fillcolor=0)

    r = rng.random()
    if r < 0.25:
        img = img.filter(ImageFilter.MaxFilter(5))
    elif r < 0.4:
        img = img.filter(ImageFilter.MinFilter(3))

    return img


def add_noise(img, rng):
    arr = img.astype(np.float32)

    if rng.random() < 0.3:
        radius = rng.uniform(0.3, 0.8)
        arr = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)

    if rng.random() < 0.7:
        arr += rng.normal(0, rng.uniform(2, 12), arr.shape)

    if rng.random() < 0.4:
        for _ in range(rng.integers(1, 4)):
            y, x = rng.integers(0, arr.shape[0], 2)
            arr[y, x] = rng.uniform(80, 255)

    return np.clip(arr, 0, 255).astype(np.uint8)



def build_dataset(examples, rng, n_augmented):
    images, labels = [], []
    for path, digit in examples:
        img = ImageOps.invert(Image.open(path).convert("L"))
        images.append(normalize_img(img))
        labels.append(digit)

        for _ in range(n_augmented):
            arr = normalize_img(augment_img(img, rng), box=int(rng.integers(17, 23)))
            if arr.max() == 0:
                continue

            images.append(arr)
            labels.append(digit)

    for _ in range(len(images) // 9):
        images.append(np.zeros((28, 28), dtype=np.uint8))
        labels.append(0)

    images = [add_noise(img, rng) for img in images]

    x = np.stack(images)
    y = np.array(labels, dtype=np.uint8)
    permutation = rng.permutation(len(x))

    return x[permutation], y[permutation]


def write_idx(prefix, x, y):
    x = np.ascontiguousarray(x, dtype=np.uint8)

    with open(f"{prefix}-images.idx3-ubyte", "wb") as f:
        f.write(struct.pack(">IIII", 2051, len(x), 28, 28))
        f.write(x.tobytes())

    with open(f"{prefix}-labels.idx1-ubyte", "wb") as f:
        f.write(struct.pack(">II", 2049, len(y)))
        f.write(y.astype(np.uint8).tobytes())


if __name__ == "__main__":
    rng = np.random.default_rng(8945)
    examples = load_chars74k("data/Chars74K/English/Fnt")
    order = rng.permutation(len(examples))

    n_test = int(0.15 * len(examples))
    test_examples = [examples[i] for i in order[:n_test]]
    train_examples = [examples[i] for i in order[n_test:]]

    x_train, y_train = build_dataset(train_examples, rng, 2)
    x_test, y_test = build_dataset(test_examples, rng, 0)

    write_idx("data/train", x_train, y_train)
    write_idx("data/test", x_test, y_test)
