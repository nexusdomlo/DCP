# Deep Closest Point

## Prerequisites 
PyTorch>=1.0: https://pytorch.org

scipy>=1.2 

numpy

h5py

tqdm

TensorboardX: https://github.com/lanpa/tensorboardX

## Training

### DCP-v1

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd

### DCP-v2

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd

## Testing

### DCP-v1

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval

or 

python main.py --exp_name=dcp_v1 --model=dcp --emb_nn=dgcnn --pointer=identity --head=svd --eval --model_path=xx/yy

### DCP-v2

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --eval

or 

python main.py --exp_name=dcp_v2 --model=dcp --emb_nn=dgcnn --pointer=transformer --head=svd --eval --model_path=xx/yy

where xx/yy is the pretrained model

## Citation
Please cite this paper if you want to use it in your work,

	@InProceedings{Wang_2019_ICCV,
	  title={Deep Closest Point: Learning Representations for Point Cloud Registration},
	  author={Wang, Yue and Solomon, Justin M.},
	  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
	  month = {October},
	  year={2019}
	}

## 自定义运行方式
python user-demo.py --checkpoint pretrained/dcp_v2.t7 --src "C:\Abandon\PCD_Data\data_2_cut.pcd" --tgt "C:\Abandon\PCD_Data\data_2.pcd" --npoints 16384
或者
python main_other.py #这个程序是用gemini2.5pro生成的，直接修改代码中的src 和 tgt的路径即可用于测试你的pcd数据，通过修改npoints的点数，来达到更好的效果，推荐是16384个点
或者
python dcp_icp_pipeline.py --src C:\Abandon\PCD_Data\data_2_cut.pcd --tgt C:\Abandon\PCD_Data\data_2_cut_transformed.pcd --dcp_checkpoint C:\Abandon\Code\Python\DCP\pretrained\dcp_v2.t7 
## License
MIT License
