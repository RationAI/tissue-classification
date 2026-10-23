#!/bin/bash
# Convert the single-level anonymized linkage slides into pyramidal TIFFs so
# xOpat loads them quickly and sharply. Run from ~/musica/backgrounds on the
# e-INFRA Jupyter; one slide at a time, the pod has a 32 GiB memory limit.
for s in SLD-ea951570 SLD-62fda854 SLD-60257199 SLD-bad3116b SLD-cd6665d8 SLD-92b035e0 SLD-06b42b6f SLD-b0c952af SLD-0b6d9611 SLD-f25a3c68; do
  [ -f $s.tiff ] && continue
  if VIPS_CONCURRENCY=4 vips tiffsave "/mnt/data/Public/MUG/linkage/$s.svs[rgb]" $s.tiff.part --tile --tile-width 512 --tile-height 512 --pyramid --compression jpeg --Q 90 --bigtiff --depth onetile; then mv $s.tiff.part $s.tiff; echo "ok $s $(date +%T)"; else echo "FAIL $s"; fi
done
echo DONE
