cd $(dirname $0)
cd ../code
pretrained_model=Salesforce/codet5-base
output_dir=../models/Data_refined_model/
CUDA_VISIBLE_DEVICES=0 python run_codet5.py \
  --do_train \
  --do_eval \
  --do_test \
  --model_type codet5 \
  --model_name_or_path $pretrained_model \
  --train_filename ../data/Data_refined/original.txt,../data/Data_refined/refined.txt \
  --dev_filename ../data/Data_refined/original.txt,../data/Data_refined/refined.txt \
  --test_filename ../data/Data_refined/test.txt,../data/Data_refined/test.txt \
  --output_dir $output_dir \
  --max_source_length 512 \
  --max_target_length 512 \
  --beam_size 6 \
  --train_batch_size 6 \
  --eval_batch_size 6 \
  --train_steps 6500 \
  --eval_steps 6500
