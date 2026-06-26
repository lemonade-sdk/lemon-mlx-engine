// Does mx::slice_update donate (in-place) or copy? Compare the buffer pointer
// before/after when the input is the sole owner (std::move pattern).
#include <mlx/mlx.h>
#include <cstdio>
namespace mx = mlx::core;
int main() {
  mx::set_default_device(mx::Device::gpu);
  int B=1,H=2,CAP=512,D=256, slot=5;
  auto buf = mx::zeros({B,H,CAP,D}, mx::float32);
  mx::eval(buf);
  void* p0 = (void*)buf.data<float>();
  auto pos = mx::array({slot}, {1}, mx::int32);
  // sole-owner update (mirrors update_at_pos after std::move)
  auto k = std::move(buf);
  auto out = mx::slice_update(k, mx::full({B,H,1,D},3.0f,mx::float32), pos, std::vector<int>{2});
  mx::eval(out);
  void* p1 = (void*)out.data<float>();
  printf("before=%p after=%p  %s\n", p0, p1,
         p0==p1 ? "DONATED (in-place)" : "COPIED (new buffer)");
  return 0;
}
