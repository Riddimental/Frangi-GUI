from pygame import mixer


def startup():
   mixer.init()
   mixer.music.load('sounds/startup.mp3')
   mixer.music.play()

def loading():
   mixer.init()
   mixer.music.load('sounds/loading.mp3')
   mixer.music.play()
   
def ai():
   mixer.init()
   mixer.music.load('sounds/ai.mp3')
   mixer.music.play()
   
def noise():
   mixer.init()
   mixer.music.load('sounds/noise.mp3')
   mixer.music.play()
   
def error():
   mixer.init()
   mixer.music.load('sounds/fail.mp3')
   mixer.music.play()

def success():
   mixer.init()
   mixer.music.load('sounds/done.mp3')
   mixer.music.play()