# deploy:
# 	sudo systemctl restart matrix && sudo journalctl -f -u matrix

remote-deploy: push
	# ssh drum@breakbox.local 'sudo systemctl restart breakbox && sudo journalctl -f -u breakbox'
	ssh pi@192.168.1.243 'sudo systemctl restart breakbox && sudo journalctl -f -u breakbox'

log:
	# ssh drum@breakbox.local  'journalctl -f -u breakbox'
	ssh pi@192.168.1.243  'journalctl -f -u breakbox'


push:
	rsync -avr --delete . pi@192.168.1.243:/home/pi/breakbox || echo 1

install-service:
	sudo cp systemd/breakbox.service /etc/systemd/system && sudo systemctl daemon-reload
