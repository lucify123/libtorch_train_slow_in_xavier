#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

static std::shared_ptr<torch::jit::script::Module> pmodule_;
auto size = cv::Size(768, 512);

torch::Tensor normalize(cv::Mat &x) {
  cv::resize(x, x, size);
  cv::cvtColor(x, x, cv::COLOR_BGR2RGB);
  torch::Tensor tensor_image =
      torch::from_blob(x.data, {1, x.rows, x.cols, 3}, torch::kByte);
  tensor_image = tensor_image.permute({0, 3, 1, 2});
  tensor_image = tensor_image.toType(torch::kFloat);

  tensor_image[0][0] = tensor_image[0][0].sub_(0.485).div_(0.229);
  tensor_image[0][1] = tensor_image[0][1].sub_(0.456).div_(0.224);
  tensor_image[0][2] = tensor_image[0][2].sub_(0.406).div_(0.225);
  tensor_image = tensor_image.div(255.0);

  return tensor_image.to(device);
}

void infer(torch::Tensor &x) {
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(x); 
  torch::Tensor out_tensor = pmodule_->forward(inputs).toTensor();
  out_tensor = out_tensor.detach();
  // out_tensor = out_tensor[0][1];
  // out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);
  // cv::Mat out(out_tensor.sizes()[0], out_tensor.sizes()[1], CV_8U,
  //             out_tensor.data_ptr());
}

void train(torch::Tensor &x) {
  std::vector<at::Tensor> parameters;
  for (const auto &p : pmodule_->parameters()) {
    parameters.emplace_back(p);
  }

  torch::optim::Adam optimizer(parameters,
                               torch::optim::AdamOptions(pow(0.1, 3)));
  optimizer.zero_grad();
  at::Tensor r = pmodule_->forward({x}).toTensor();
  at::Tensor y = r - 1e-2;
  auto tloss = torch::nn::BCELoss()(r, y.detach());
  tloss.backward();

  optimizer.step();
}

int main() {
  auto x = cv::imread("4x.jpg");
  torch::Tensor xx= normalize(x);
  pmodule_ = std::make_shared<torch::jit::script::Module>(
      torch::jit::load("init.smart", device));
  std::cout << "warming up,pls wait..." << std::endl;
  for (int i = 0; i < 10; i++) {
    infer(xx);
    train(xx);
  }
  auto start = std::chrono::system_clock::now();
  for (int i = 0; i < 50; i++) {
    infer(xx);
  }
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double t = double(duration.count()) * std::chrono::microseconds::period::num /
             std::chrono::microseconds::period::den;
  std::cout << "infer cost " << t << " s" << std::endl;
  for (int i = 0; i < 50; i++) {
    train(xx);
  }
  auto e = std::chrono::system_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(e - end);
  t = double(duration.count()) * std::chrono::microseconds::period::num /
      std::chrono::microseconds::period::den;
  std::cout << "train cost " << t << " s" << std::endl;
  return 0;
}