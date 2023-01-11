'''
Ref:
https://ithelp.ithome.com.tw/articles/10280087?sc=pt  : sendmessage
https://api.imgur.com/oauth2/addclient : upload img
https://www.learncodewithmike.com/2020/06/python-line-bot.html : connect python and line bot
'''


from django.shortcuts import render
from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from linebot.models.send_messages import ImageSendMessage
from linebot import LineBotApi, WebhookParser
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextSendMessage
import os
import random
import pyimgur
line_bot_api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)
parser = WebhookParser(settings.LINE_CHANNEL_SECRET)
def glucose_graph(client_id, imgpath):
	im = pyimgur.Imgur(client_id)
	upload_image = im.upload_image(imgpath, title="Uploaded with PyImgur")
	return upload_image.link


@csrf_exempt
def callback(request):

    if request.method == 'POST':
        signature = request.META['HTTP_X_LINE_SIGNATURE']
        body = request.body.decode('utf-8')

        try:
            events = parser.parse(body, signature)  # 傳入的事件
        except InvalidSignatureError:
            return HttpResponseForbidden()
        except LineBotApiError:
            return HttpResponseBadRequest()
        finger_guessing_game = ['剪刀', '石頭', '布']
        for event in events:
            k = random.randint(0, 2)
            UserId = event.source.user_id
            name = line_bot_api.get_profile(UserId).display_name
            status = line_bot_api.get_profile(UserId).status_message
            if isinstance(event, MessageEvent):  # 如果有訊息事件
                if event.message.text == '猜拳':
                    line_bot_api.reply_message(
                        event.reply_token, TextSendMessage(text=finger_guessing_game[k]))
                elif event.message.text[:4] == "幫我創造":
                    try:
                        imagination = event.message.text[5:]
                        if os.path.isfile(f"E:/AI_draw/stable-diffusion-main/outputs/txt2img-samples/{imagination}.png"):
                            local_save = f'E:/AI_draw/stable-diffusion-main/outputs/txt2img-samples/{imagination}.png'
                            img_url = glucose_graph('abab256a1ee90fe', local_save)
                        else:
                            os.system(f'E: && cd AI_draw/stable-diffusion-main/ && conda activate ldm && python optimizedSD/optimized_txt2img.py --prompt "{imagination}" --H 512 --W 512 --seed 27 --n_iter 1 --n_samples 1 --ddim_steps 50 --precision full')
                            local_save = f'E:/AI_draw/stable-diffusion-main/outputs/txt2img-samples/{imagination}.png'
                            img_url = glucose_graph('abab256a1ee90fe', local_save)
                        line_bot_api.reply_message(event.reply_token, ImageSendMessage(original_content_url=img_url, preview_image_url=img_url))
                    except:
                        local_save = f'E:/AI_draw/stable-diffusion-main/outputs/txt2img-samples/{imagination}.png'
                        img_url = glucose_graph('abab256a1ee90fe', local_save)
                        line_bot_api.reply_message(event.reply_token, ImageSendMessage(original_content_url=img_url, preview_image_url=img_url))
                else:
                    line_bot_api.reply_message(  # 回復傳入的訊息文字
                        event.reply_token,
                        TextSendMessage(
                            text=f"你好 {name} \n你的userid是{UserId}\n 你的狀態消息為 {status}")
                    )
        return HttpResponse()
    else:
        return HttpResponseBadRequest()
