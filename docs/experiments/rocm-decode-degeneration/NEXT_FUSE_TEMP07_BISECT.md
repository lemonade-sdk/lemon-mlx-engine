# Bisect complete → see RESOLUTION

**Superseded by:** [`RESOLUTION_FUSE_TEMP07.md`](./RESOLUTION_FUSE_TEMP07.md)

Isolation at tip `710135e` finished with three cells:

| Cell | Result |
|------|--------|
| FUSE temp=0 think | **PASS** |
| FUSE temp=0.7 think | **FAIL** (mid-T5 thrash) |
| NOFUSE temp=0.7 think | **PASS** |

**Conclusion:** fuse × sampling @0.7. Fuse stays opt-in (`MLX_ENABLE_QUANT_FUSE`); B2/B3 not required. Full evidence and product disposition are in the resolution doc.
