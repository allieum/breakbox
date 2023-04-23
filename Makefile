# deploy:
# 	sudo systemctl restart matrix && sudo journalctl -f -u matrix

# remote-deploy: push
# 	ssh drum@breakbox.local 'sudo systemctl restart matrix && sudo journalctl -f -u matrix'

push:
	rsync -avr /home/rachel/allie/breakbox drum@breakbox.local:/home/drum/

# install-service:
# 	sudo cp systemd/matrix.service /etc/systemd/system && sudo systemctl daemon-reload
