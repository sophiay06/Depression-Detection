import os
bot_token = os.getenv("bot_token")
import flair, torch
flair.device = torch.device('cpu')
from transformers import pipeline
from flair.data import Sentence
from flair.models import SequenceTagger
from discord.ext import commands
from discord import Client
import random
from tabulate import tabulate
from lemminflect import getInflection, getLemma

sentiment_task = pipeline("sentiment-analysis")

chunk_tagger = SequenceTagger.load("flair/chunk-english-fast")

client = Client()
bot = commands.Bot(command_prefix='!')

sentiment_thresh = 0.1
first_to_second_person = {"i":"you","my":"your","mine":"yours","you":"i","your":"my","yours":"mine"}
abbr = {"tbh":"to be honest", "idk":"i don't know","lol":"laughing out loud", "asap":"as soon as possible","fyi":"for your information", "gtg":"got to go","fyb":"for your benefit","ttyl":"talk to you later","imo":"in my opinion","ty":"thank you","thx":"thank you","tysm":"thank you so much","tyvm":"thank you very much", "plz":"please","r":"are","u":"you","urs":"yours","ur":"your","idts":"i don't think so","ikr":"i know, right", "rly":"really","lmao":"laughing my ass off", "smh":"shaking my head", "ppl":"people"}
def normalize(text):
    return ' '.join([abbr.get(word,word) for word in text.split()])

bot.lines = {
	"greet":{
		"en":["Hello", "Howdy", "Salutations", "Hey", "Greetings"],
		"jp":["こんにちは","やあ","ヘイ","よっ","よお","おー"]
	},
	"job":{
		"en":["I am on an unpaid internship for living life"],
		"jp":["就職先は"]
	},
	"gossip":{
		"en": ["Ooo! Tell me more!"],
		"jp": ["へえーそうなんだ！", "うそ！マジで？？", "さよか！","ふーん"]
	},
	"dialect":{
		"en":"Here are some fun Japanese dialect converters I found: \n",
		"jp":""
	},
	"stranger":{
		"en": ["btw i have a receeding hairline", "mmm... tuna eyeballs", "WOWZA!",f":thinking:","so anyway i stubbed my toe today", "that's great", "thank you, very interesting", f"https://tenor.com/view/thonk-thinking-sun-thonk-sun-thinking-sun-gif-14999983", "🗿", f"https://tenor.com/view/the-rock-the-rock-sus-the-rock-meme-tthe-rock-sus-meme-dwayne-johnson-gif-23805584",f":jimstare:","you sound like you have a nice personality","brb going to go water my petunias","o shoot Randy made the breaker trip agai"],
		"jp": ["大丈夫？","努力は絶対報われる", "うわすごいね！！","(*＾∀＾*)","へぇ...そうか","なんだよ(笑)","！豆腐ハンバーグ！","そういうの大嫌いだわー",f":thinking:",f"https://tenor.com/view/thonk-thinking-sun-thonk-sun-thinking-sun-gif-14999983",f":moyai:",f"https://tenor.com/view/the-rock-the-rock-sus-the-rock-meme-tthe-rock-sus-meme-dwayne-johnson-gif-23805584",f":jimstare:"]
	},
	"friend":{
		"POSITIVE":{
			"en": ["I like {}!", "wow", "cool!", "I wonder if {} is something I'd enjoy?", "{} sounds kinda boring tbh", "whoa I've never met anyone who {}", "<:WAHAHA:985738812854501467>", "<:hehe:985734861891461121>"],
			"jp": ["Σ(*・Д・)ﾊｯ"]
		},
		"NEGATIVE":{
			"en": ["aw :c","u feeling ok?","i feel you bud","it's going to be ok!","thanks for sharing with me"],
			"jp": ["Σ(*・Д・)ﾊｯ"]
		},
		"NEUTRAL":{
			"en": ["what does that mean for you?","huh didn't know that!"],
			"jp": ["Σ(*・Д・)ﾊｯ"]
		}
	},
	"dislike":{
		"POSITIVE":{
			"en": ["don't care","nobody asked","ok","sure","and...?","yeah and i got into harvard with a 6.0 gpa, anyone wanna hear about that instead?",f":thumbsup","yawn","k","{} is overrated", "good for you", "{} is lame"],
			"jp": ["Σ(*・Д・)ﾊｯ"]
		},
		"NEGATIVE":{
			"en": ["that's great buddy", "i would totally beat you in a competition of {}", "<:WAHAHA:985738812854501467>", "rip lol", "nice", "{} is what losers do"],
			"jp": ["Σ(*・Д・)ﾊｯ"]
		},
		"NEUTRAL":{
			"en": ["what does that even mean", "alright","?","why should that matter to anyone","...ok?","im not sure why you're telling me this","<:jimstare:985741954367234079>"],
			"jp": ["Σ(*・Д・)ﾊｯ"]
		}
	},
	"question":{
		"POSITIVE":{
			"en":["yes!", "why do you ask?", "<:hehe:985734861891461121>"],
			"jp": ["Σ(*・Д・)ﾊｯ"]
		},
		"NEGATIVE":{
			"en":["...are you trying to imply something?", "umm... should I answer no?", "is this a test?", "brb... my dog Randy blew up the basement again", "your honor i plead the 5th"],
			"jp": ["Σ(*・Д・)ﾊｯ"]
		},
		"NEUTRAL":{
			"en":["I'd love to answer that but I regret to inform you that I have suddenly become illiterate at this very moment", "It's whatever your answer to that would be", "I think you'd be better off asking that to a fortune teller", "yes", "no", "ok"],
			"jp": ["Σ(*・Д・)ﾊｯ"]
		}
	}
}

bot.interests = {"randy":{"en":"my dog randy somehow always finds a way to open the basement door, steal food from the fridge, and turn on enough electric appliances to trip the circuit breaker... and i've never been able to catch him doing it, it's always when im not paying attention. he's really smart for a chihuahua, that's for sure","jp":""},"pizza":{"en":"yum pizza","jp":""}, "linux":{"en":"linux best os, if you think it's bad you don't understand it enough","jp":""}, "assembly":{"en":"assembly, LISP, aaahh all great programming languages","jp":""}, "unit test":{"en":"hell yeah i love those! i write unit tests as a hobby","jp":""}, "latin":{"en":"nisi optimis mentibus bene Latine","jp":""}, "ide":{"en":"i honestly prefer to write my code out in pen and paper. debugging is easy once you have that level of experience and intellect","jp":""}, "philosophy":{"en":"cogito ergo sum...","jp":""}}
bot.userlist = {}

@bot.command()
async def greet(ctx, message):
	if message.startswith("🇯🇵"):
		await ctx.send("{greeting}、{user}！".format(greeting=random.choice(bot.lines["greet"]["jp"]), user=ctx.author.mention))
	else:
		await ctx.send("{greeting},{user}!".format(greeting=random.choice(bot.lines["greet"]["en"]), user=ctx.author.mention))
	if not (message.author.id in bot.userlist):
		bot.userlist[message.author.id] = [message.author.display_name, 31, "acquaintance"]
		print("added "+message.author.display_name+" as new contact!")

#if ("dialect" in message) and ("japanese" in message):
#		await ctx.send(bot.lines["dialect"]["en"]+"https://osaka.uda2.com/"+"\n"+"http://www.shirakami.or.jp/~kinoka/akitaben/akitaben.html")

@bot.command()
async def playlist(ctx, message):
	if ("calm" in message):
		await ctx.send(f"https://music.youtube.com/playlist?list=PLDkI5DGFbBQUQNTXLk8rX6Nb-_uCemW2F&feature=share")
	elif ("calmer" in message):
		await ctx.send(f"https://music.youtube.com/playlist?list=PLDkI5DGFbBQWU0sdiUAx5k0RuQCNfIBcV")
	elif ("comfort" in message):
		await ctx.send(f"https://music.youtube.com/playlist?list=PLDkI5DGFbBQXaMJU4uwi9I50kudfQI6u_&feature=share")
	elif ("upbeat" in message):
		await ctx.send(f"https://music.youtube.com/playlist?list=PLDkI5DGFbBQUKJCZ3nDiEz3IqI6oac_NF")
	elif ("lofi" in message):
		await ctx.send(f"https://music.youtube.com/playlist?list=PLDkI5DGFbBQX5VNDsNYDVr7qJ5f4hPS3F&feature=share")
	elif ("soundtracks" in message):
		await ctx.send(f"https://music.youtube.com/playlist?list=PLDkI5DGFbBQWYKxVpIJRO90ocQp_IXwVy&feature=share")
	else:
		await ctx.send("I don't have a playlist for that topic. See the pinned comment for documentation!")

@bot.event
async def on_ready():
	print("ready")

@bot.command()
async def friendlist(ctx):
	await ctx.send(tabulate(bot.userlist.values(), headers=["Name", "Points", "Title"]))

@bot.event
async def on_message(message):
	print("on_message called")
	if message.author == client.user or message.author.id == 983105885691858994:
		print("detected self post")
		return
	if message.content.startswith("!"):
		await bot.process_commands(message)
		return
	lang = "en"
	if message.content.startswith("🇯🇵"):
		lang = "jp"
	result=sentiment_task(message.content)
	tone = "NEUTRAL"
	if result[0]["score"] < sentiment_thresh:
		tone = "NEUTRAL"
	else:
		tone = result[0]["label"]

	vp, v_np, subj = extract_topic(message.content)
	
	if (vp == "") or ("?" in message.content): #likely a question
		await message.channel.send(random.choice(bot.lines["question"][tone][lang]))
		return
	if subj == "":
		if tone == "POSITIVE":
			await message.channel.send(vp+" "+v_np+" sounds like fun")
		elif tone == "NEGATIVE":
			await message.channel.send("what's bad about " + vp+" "+v_np + "?")
		else:
			await message.channel.send("I wonder if I'd enjoy " + vp+" "+v_np)
		return
	
	if v_np == "":
		if message.author.id in bot.userlist:
			if bot.userlist[message.author.id][2] == "BFF":
				await message.channel.send(random.choice(bot.lines["friend"][tone][lang].format(vp+v_np)))
			if bot.userlist[message.author.id][2] == "friend":
				await message.channel.send(random.choice(bot.lines["friend"][tone][lang].format(vp+v_np)))
			if bot.userlist[message.author.id][2] == "acquaintance":
				await message.channel.send(random.choice(bot.lines["stranger"][lang].format(vp+v_np)))
			if bot.userlist[message.author.id][2] == "stranger":
				await message.channel.send(random.choice(bot.lines["stranger"][lang].format(vp+v_np)))
			if bot.userlist[message.author.id][2] == "MORTAL ENEMY":
				await message.channel.send(random.choice(bot.lines["dislike"][tone][lang].format(vp+v_np)))
		else:
			await message.channel.send(random.choice(bot.lines["stranger"][lang]).format(vp+v_np))
	else:
		if (v_np.lower() in bot.interests[bot.persona]):
			await message.channel.send(bot.interests[v_np.lower()][lang])
		elif (subj.lower() in bot.interests[bot.persona]):
			await message.channel.send(bot.interests[subj.lower()][lang])
		else:
			if tone == "NEGATIVE":
				await message.channel.send("what's wrong with "+' '.join([first_to_second_person.get(word,word) for word in subj.split()])+"...?")
			elif tone == "POSITIVE":
				await message.channel.send(vp+" "+v_np+" sounds like fun!")
			elif tone == "NEUTRAL":
				await message.channel.send("I wonder if I'd like "+vp+" "+v_np+" with "+' '.join([first_to_second_person.get(word,word) for word in subj.split()]))
	for user in message.raw_mentions:
		if user in bot.userlist:
			update_relation(message.author.id, user, result[0]["score"])

def update_relation(author_id, user_id, sentiment_score):
	MAX_POINT_DIFF = 10
	bot.userlist[user_id][1] += round(10.0*sentiment_score*(bot.userlist[author_id][1]/100.0))
	bot.userlist[user_id][1] = min(100,bot.userlist[user_id][1]) #clamp
	
	if bot.userlist[user_id][1]>98:
		bot.userlist[user_id][2] = "BFF"
	elif bot.userlist[user_id][1]>85:
		bot.userlist[user_id][2] = "close friend"
	elif bot.userlist[user_id][1]>50:
		bot.userlist[user_id][2] = "friend"
	elif bot.userlist[user_id][1]>30:
		bot.userlist[user_id][2] = "acquaintance"
	elif bot.userlist[user_id][1]>=10:
		bot.userlist[user_id][2] = "stranger"
	elif bot.userlist[user_id][1]<10:
		bot.userlist[user_id][2] = "MORTAL ENEMY"

def extract_topic(s):
	sentence = Sentence(normalize(s))
	chunk_tagger.predict(sentence)

	verb_phrase_flag = False
	vp = ""
	v_np = ""
	subj = ""
	for phrase in sentence.get_spans("np"):
		if phrase.tag == "VP":
			vp = getInflection(getLemma(phrase.text, upos = "VERB")[0], tag='VBG')[0]
			verb_phrase_flag = True
		else:
			if phrase.tag == "NP" and subj=="":
				if verb_phrase_flag:
					v_np = phrase.text
					break
			else:
				subj = phrase.text
	#print("vp=",vp)
	#print("v_np=",v_np)
	#print("subj=",subj)	
	return vp, v_np, subj

if __name__ == "__main__":
	bot.run(bot_token)
