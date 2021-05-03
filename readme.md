Run development server with: 
`FLASK_APP=src/server/server.py  FLASK_ENV=development python -m flask run`

run gunicorn with:
`gunicorn --bind unix:server.sock -m 007 src.server.wsgi:app`

because of versioning issues, torch packages need to be manually installed in the venv with: 

`python -m pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html`

Spectrogram statistics across the whole dataset:
Mean=879351519686/8453355264=104.02396352970787
Min=0; Max=156
variation=6068880539648.0/8453355264=717.9255981445312; stdev=26.794133651688224