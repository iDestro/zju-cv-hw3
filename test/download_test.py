import wget

url = 'https://bgmnist.oss-cn-hangzhou.aliyuncs.com/test_set_target_binary.npy'
wget.download(url, '../download/test_set_target_binary.npy')
