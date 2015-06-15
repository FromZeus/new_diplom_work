from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle
from kivy.uix.textinput import TextInput

class NeuroLayout(FloatLayout):
  def hulk_smash(self):
    print "LOL!"

class NeuronetApp(App):
  icon = 'attracto.png'
  title = 'Attracto'

  def build(self):
    return NeuroLayout()

if __name__ == '__main__':
    NeuronetApp().run()