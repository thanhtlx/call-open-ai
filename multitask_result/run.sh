pip install -r requirements.txt
#python3 -W ignore finetuning.py --batch_size 16·
python3 -W ignore gen.py --batch_size 4
zip -r result.zip tmp test.codellama.reload.output
