# deploy:
# 	sudo systemctl restart matrix && sudo journalctl -f -u matrix

remote-deploy: push
	ssh drum@breakbox.local 'sudo systemctl restart breakbox && sudo journalctl -f -u breakbox'

log:
	ssh drum@breakbox.local  'sudo journalctl -f -u breakbox'

push:
	rsync -avr --exclude-from 'rsync.exclude' `pwd` drum@breakbox.local:/home/drum/
	rsync -avr --delete `pwd`/src/samples drum@breakbox.local:/home/drum/breakbox/src/

install-service:
	sudo cp systemd/breakbox.service /etc/systemd/system && sudo systemctl daemon-reload
