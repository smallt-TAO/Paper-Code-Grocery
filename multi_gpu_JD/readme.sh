cancer_classifier.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=ps --task_index=0
 
cancer_classifier.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=ps --task_index=1
 
cancer_classifier.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=worker --task_index=0
 
cancer_classifier.py --ps_hosts=127.0.0.1:2222,127.0.0.1:2223 --worker_hosts=127.0.0.1:2224,127.0.0.1:2225 --job_name=worker --task_index=1
