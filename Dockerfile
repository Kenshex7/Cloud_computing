FROM python

RUN apt-get update

#########
###MPI###
#########
RUN yes | apt install mpich
RUN yes | pip3 install mpi4py

ADD example.py /mpi/example.py

#########
###SSH###
#########

RUN apt-get install -y openssh-server
RUN mkdir /var/run/sshd

RUN echo 'root:root' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

WORKDIR /mpi

RUN mkdir -p /root/.ssh/
ADD mpi_cluster.pub mpi_cluster.pub
ADD mpi_cluster mpi_cluster

RUN cat mpi_cluster.pub >> /root/.ssh/authorized_keys
RUN cp mpi_cluster /root/.ssh/id_rsa
RUN cp mpi_cluster.pub /root/.ssh/id_rsa.pub

RUN chmod 600 /root/.ssh/id_rsa

# Needed to bypass checking the host
COPY ssh_config /etc/ssh/ssh_config

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
RUN service ssh restart
CMD ["/usr/sbin/sshd", "-D"]

RUN yes | pip3 install numpy
RUN yes | pip3 install opencv-python
RUN yes | pip3 install google-cloud-firestore
RUN yes | pip3 install pathlib
RUN yes | pip3 install pytest-shutil
RUN yes | pip3 install google-cloud-storage
