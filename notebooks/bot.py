import telebot
import pickle
import re

TOKEN = 'my_token'
bot = telebot.TeleBot(TOKEN)


def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]+", '', text)
    text = text.replace('"', '')
    return text


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id,
                     f'Hi, {message.from_user.first_name}! Send me text and i will predict rating')


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    model = load_model('model.pkl')
    user_text = message.text
    preprocessed_text = preprocess_text(user_text)
    prediction = model.predict([preprocessed_text])[0]
    bot.send_message(message.chat.id, f'Predicted rating: {prediction}')


if __name__ == '__main__':
    bot.polling(none_stop=True)
