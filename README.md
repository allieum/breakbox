# Running on Docker

## Mac OS

<https://stackoverflow.com/questions/40136606/how-to-expose-audio-from-docker-container-to-a-mac>

```
brew install pulseaudio
pulseaudio --load=module-native-protocol-tcp --exit-idle-time=-1 --daemon
```
