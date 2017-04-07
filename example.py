#! -*- coding: utf-8 -*-

from tweet2vec.interface import Twee2vecInterface, PostRecordObject, PredictionRecordObject, InputDataset, ModelObject
from tweet2vec import settings_char
from tweet2vec.logger import logger
import os
import shutil

"""For making training/validation/test data, you put your data with PostRecordObject class"""

seq_training_record = [
    PostRecordObject(post_id=1, post_text="""The Hong Kong Disneyland Resort is a resort built and owned by Hongkong International Theme Parks Limited, a joint venture of the Government of Hong Kong and The Walt Disney Company in Hong Kong on reclaimed land beside Penny's Bay,[1] at the northeastern tip of Lantau Island, approximately two kilometres from Discovery Bay.""", post_label="Amusement parks"),
    PostRecordObject(post_id=2, post_text="""Universal Studios Japan , located in Osaka, is one of four Universal Studios theme parks, owned and operated by USJ Co., Ltd. which is wholly owned by NBCUniversal (as of 2017).""", post_label="Amusement parks"),
    PostRecordObject(post_id=3, post_text="""Miracle Strip at Pier Park was an amusement park in Panama City Beach, Florida, owned by Miracle Strip Carousel, LLC.""", post_label="Amusement parks"),
    PostRecordObject(post_id=3, post_text="""Ferrari World Abu Dhabi is an amusement park located on Yas Island in Abu Dhabi, United Arab Emirates. It is the first Ferrari-branded theme park and has the record for the largest space frame structure ever built.""", post_label="Amusement parks"),
    PostRecordObject(post_id=4, post_text="""Sega World Sydney was an indoor high-tech amusement park that operated for almost four years, in Sydney, Australia.""", post_label="Amusement parks"),
    PostRecordObject(post_id=5, post_text="""The Big Banana is a tourist attraction and amusement park in the city of Coffs Harbour, New South Wales, Australia.""", post_label="Amusement parks"),
    PostRecordObject(post_id=6, post_text="""The Matrix is a 1999 science fiction film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano.""", post_label="Movie"),
    PostRecordObject(post_id=7, post_text="""Jurassic Park is an American media franchise centering on a disastrous attempt to create a theme park of cloned dinosaurs. It began in 1990 when Universal Studios bought the rights to the novel by Michael Crichton before it was even published.""", post_label="Movie"),
    PostRecordObject(post_id=8, post_text="""The Last Samurai is a 2003 American epic historical war film directed and co-produced by Edward Zwick, who also co-wrote the screenplay with John Logan and Marshall Herskovitz.""", post_label="Movie"),
    PostRecordObject(post_id=9, post_text="""Forrest Gump is a 1994 American comedy-drama film based on the 1986 novel of the same name by Winston Groom. The film was directed by Robert Zemeckis and stars Tom Hanks, Robin Wright, Gary Sinise, Mykelti Williamson, and Sally Field.""", post_label="Movie"),
    PostRecordObject(post_id=10, post_text="""Full Metal Jacket is a 1987 British-American war film directed and produced by Stanley Kubrick. The screenplay by Kubrick, Michael Herr, and Gustav Hasford was based on Hasford's novel The Short-Timers (1979).""", post_label="Movie"),
    PostRecordObject(post_id=11, post_text="""Vincent Willem van Gogh was a Dutch Post-Impressionist painter who is among the most famous and influential figures in the history of Western art.""", post_label="Painter"),
    PostRecordObject(post_id=12, post_text="""Eugène Henri Paul Gauguin was a French post-Impressionist artist. Underappreciated until after his death, Gauguin is now recognized for his experimental use of color and synthetist style that were distinctly different from Impressionism.""", post_label="Painter"),
    PostRecordObject(post_id=13, post_text="""Paul Cézanne was a French artist and Post-Impressionist painter whose work laid the foundations of the transition from the 19th-century conception of artistic endeavour to a new and radically different world of art in the 20th century.""", post_label="Painter"),
    PostRecordObject(post_id=14, post_text="""Henri-Émile-Benoît Matisse  was a French artist, known for both his use of colour and his fluid and original draughtsmanship.""", post_label="Painter"),
    PostRecordObject(post_id=15, post_text="""Kanō Naganobu  was a Japanese painter of the Kanō school. Naganobu was the youngest brother of the Kanō school's head, Kanō Eitoku. Naganobu completed numerous commissions for the court in Kyoto, including at the Imperial Palace, and started his own line of the Kanō school.""", post_label="Painter"),
    PostRecordObject(post_id=16, post_text="""Lyon or (more archaically) Lyons is a city in east-central France, in the Auvergne-Rhône-Alpes region, about 470 km (292 mi) from Paris and 320 km (199 mi) from Marseille. Inhabitants of the city are called Lyonnais.""", post_label="City"),
    PostRecordObject(post_id=17, post_text="""Antananarivo , then temporarily French Tananarive , also known by its French colonial shorthand form Tana, is the capital and largest city of Madagascar.""", post_label="City"),
    PostRecordObject(post_id=18, post_text="""Bangkok is the capital and most populous city of Thailand. It is known in Thai as Krung Thep Maha Nakhon  or simply Krung Thep. The city occupies 1,568.7 square kilometres (605.7 sq mi) in the Chao Phraya River delta in Central Thailand, and has a population of over 8 million, or 12.6 percent of the country's population. Over 14 million people (22.2 percent) live within the surrounding Bangkok Metropolitan Region, making Bangkok an extreme primate city, significantly dwarfing Thailand's other urban centres in terms of importance.""", post_label="City"),
    PostRecordObject(post_id=19, post_text="""Nagoya is the largest city in the Chūbu region of Japan. It is Japan's third-largest incorporated city and the fourth most populous urban area.""", post_label="City"),
    PostRecordObject(post_id=20, post_text="""Milan is a city in Italy, capital of the Lombardy region, and the most populous metropolitan area and the second most populous comune in Italy.""", post_label="City")
]

seq_validation_record = [
    PostRecordObject(post_id=100, post_text="""Warner Bros. Movie World (more commonly referred to as Movie World) is a popular movie related theme park on the Gold Coast, Queensland, Australia. It is owned and operated by Village Roadshow since the take over from Time Warner and is the only movie related park in Australia. It opened on 3 June 1991.""", post_label="Amusement parks"),
    PostRecordObject(post_id=102, post_text="""Parque Warner Madrid (previously known as Warner Bros. Movie World Madrid and Warner Bros. Park) is a theme park located 25 km southeast of Madrid, Spain, in the municipality of San Martín de la Vega. The park opened on April 6, 2002, under the management of the Six Flags chain, with a 5% ownership share held by Time Warner.""", post_label="Amusement parks"),
    PostRecordObject(post_id=103, post_text="""Speed is a 1922 American action film serial directed by George B. Seitz.""", post_label="Movie"),
    PostRecordObject(post_id=104, post_text="""Die Hard is a 1988 American action film directed by John McTiernan and written by Steven E. de Souza and Jeb Stuart.""", post_label="Movie"),
    PostRecordObject(post_id=105, post_text="""Pablo Ruiz y Picasso, also known as Pablo Picasso was a Spanish painter, sculptor, printmaker, ceramicist, stage designer, poet and playwright who spent most of his adult life in France.""", post_label="Painter"),
    PostRecordObject(post_id=106, post_text="""Oscar-Claude Monet was a founder of French Impressionist painting, and the most consistent and prolific practitioner of the movement's philosophy of expressing one's perceptions before nature, especially as applied to plein-air landscape painting.""", post_label="Painter"),
    PostRecordObject(post_id=107, post_text="""Tehran is the capital of Iran and Tehran Province. With a population of around 9 million in the city and 16 million in the wider metropolitan area""", post_label="City"),
    PostRecordObject(post_id=108, post_text="""Abu Dhabi is the capital and the second most populous city of the United Arab Emirates, and also capital of the Emirate of Abu Dhabi, the largest of the UAE's seven emirates.""", post_label="City"),
]

seq_test_data = [
    PostRecordObject(post_id=200, post_text="""PortAventura World is an entertainment resort in the south of Catalonia, in Salou, Tarragona, Spain; on the Costa Daurada. It was built around the PortAventura theme park, which attracts around 4 million visitors per year making it the most visited theme park in Spain.""", post_label="Amusement parks"),
    PostRecordObject(post_id=201, post_text="""Melbourne's Luna Park is a historic amusement park located on the foreshore of Port Phillip Bay in St Kilda, Melbourne, Victoria.""", post_label="Amusement parks"),
    PostRecordObject(post_id=202, post_text="""Aladdin is a 1992 American animated musical fantasy film produced by Walt Disney Feature Animation and released by Walt Disney Pictures.""", post_label="Movie"),
    PostRecordObject(post_id=203, post_text="""The Lion King is a 1994 American animated epic musical film, produced by Walt Disney Feature Animation and released by Walt Disney Pictures. It is the 32nd Disney animated feature film.""", post_label="Movie"),
    PostRecordObject(post_id=204, post_text="""Johannes, Jan or Johan Vermeer was a Dutch painter who specialized in domestic interior scenes of middle-class life.""", post_label="Painter"),
    PostRecordObject(post_id=205, post_text="""Salvador Domingo Felipe Jacinto Dalí i Domènech, Marqués de Dalí de Púbol, known professionally as Salvador Dalí , was a prominent Spanish surrealist painter born in Figueres, Catalonia, Spain.""", post_label="Painter"),
    PostRecordObject(post_id=206, post_text="""The City of New York, often called New York City or simply New York, is the most populous city in the United States.""", post_label="City"),
    PostRecordObject(post_id=207, post_text="""Delhi, officially the National Capital Territory of Delhi or NCT, is a city and a union territory of India.""", post_label="City"),
]

### If your text is longer than 144, set longer value on MAX_LENGTH ###
settings_char.MAX_LENGTH = 250

### Make daaset object ###
training_dataset = InputDataset.load_from_generic_input(seq_training_record)
validation_dataset = InputDataset.load_from_generic_input(seq_validation_record)
test_dataset = InputDataset.load_from_generic_input(seq_test_data)

twee2vec_obj = Twee2vecInterface()
### directory to save trained model ###
PATH_TRAINED_MODEL = './example-model'
if not os.path.exists(PATH_TRAINED_MODEL):
    os.mkdir(PATH_TRAINED_MODEL)
### Model training. The trained model is saved under PATH_TRAINED_MODEL ###
model_object = twee2vec_obj.train(training_dataset=training_dataset,
                                  validation_dataset=validation_dataset,
                                  save_dir=PATH_TRAINED_MODEL)
#### You can load model object from disk ###
trained_model_obj = ModelObject.load_model(PATH_TRAINED_MODEL)

### Prediction ###
seq_predicted_obj = twee2vec_obj.predict(test_data=test_dataset,
                                         model_object=trained_model_obj)

for prediction_result in seq_predicted_obj:
    assert isinstance(prediction_result, PredictionRecordObject)
    print('Prediction label={} for id={} text={}'.format(
        prediction_result.prediction_label,
        prediction_result.post_id,
        prediction_result.post_text))

shutil.rmtree(PATH_TRAINED_MODEL)