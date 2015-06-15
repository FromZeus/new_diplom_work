from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle
from kivy.uix.textinput import TextInput

class TestWidget(Widget):
  pass

class TestLayout(FloatLayout):
  def hulk_smash(self):
    print self.sample_text
    print "LOL!"

class TestApp(App):
    def build(self):
      return TestLayout()
        #return Button(text='Hello World')

if __name__ == '__main__':
    TestApp().run()