# Custom Photo Layout

Use paired captures when possible:

```text
reflection/
  001.png
  002.png
  003.png
  004.png
  005.png
clean/
  001.png
  002.png
  003.png
  004.png
  005.png
```

If clean references are unavailable, put only reflection inputs under:

```text
reflection_only/
```

Only paired images should be used for PSNR, SSIM, NCC, and LMSE. Reflection-only images should be used for qualitative figures.
