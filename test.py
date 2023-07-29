
if __name__ == "__main__":
    from g2pw import G2PWConverter
    conv = G2PWConverter(model_dir="g2pW_models/models", enable_non_tradional_chinese=True,use_cuda=False,num_workers=32)
    import time
    t0 = time.time()
    texts = ["好的，您可以通过支付宝，微信或手动还款，那就不打扰了，我们会持续关注您的还款，再见，", "那您看这样行吗"]
    for text in texts:
        out = conv(text)
        res = " ".join(out[0])
        print(res)
    print(time.time()-t0)