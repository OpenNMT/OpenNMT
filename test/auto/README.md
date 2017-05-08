To launch an automatic test run:

* update in `config.yaml` paths to:
  * `log_dir` where the test log and the badge for the run will be created. The file `alllog.html` 
will be appended with link to both.
  * `tmp_dir` which is directory where the runs will be launched (erased everytime you launch again)
  * `download_dir` which is a directory where required files will be downloaded (not erased when you launch again)
* launch:
```
th test/auto/launch.lua
```

The default configuration is for run on AWS EC2 with following sequence:

  * launch AMI `ami-8463e892` EC2 p2 instance - the instance has a preconfigured installation of torch and necessary packages
  * install lua yaml library: `luarocks install yaml`
  * clone OpenNMT repository: `git clone https://www.github.com/OpenNMT/OpenNMT`
  * run the test run from OpenNMT directory
