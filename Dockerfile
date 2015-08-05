FROM texastribune/supervisor
MAINTAINER tech@texastribune.org

RUN pip install -i http://pypi.douban.com/simple/ numpy

RUN pip install -i http://pypi.douban.com/simple/ Cython
RUN apt-get update
RUN apt-get install -y python-scipy

WORKDIR /app
RUN mkdir ./leanEngine_app

ADD leanEngine_app ./leanEngine_app
ADD supervisor.conf /etc/supervisor/conf.d/
RUN pip install -i http://pypi.douban.com/simple/ -r ./leanEngine_app/requirements.txt

RUN pip install -i http://pypi.douban.com/simple/ gunicorn==19.1.1

EXPOSE 9010
