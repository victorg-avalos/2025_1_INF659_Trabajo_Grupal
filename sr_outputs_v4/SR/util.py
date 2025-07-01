from PIL import Image

# Base name (everything before the “_B/G/R.png” suffix)
base = "AisazuNihaIrarenai_000_color"

# Open each channel image (make sure they're all the same size)
b = Image.open(f"{base}_B.png").convert("L")
g = Image.open(f"{base}_G.png").convert("L")
r = Image.open(f"{base}_R.png").convert("L")

# Merge into one RGB image (Pillow expects channels in R, G, B order)
rgb = Image.merge("RGB", (r, g, b))

# Save the result
rgb.save(f"{base}_fused_color.png")

print(f"Saved fused image as {base}_fused_color.png")
