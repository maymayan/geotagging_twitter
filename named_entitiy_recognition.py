import jpype as jp

ZEMBEREK_PATH = 'zemberek/zemberek-full.jar'

jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH))

TurkishMorphology = jp.JClass('zemberek.morphology.TurkishMorphology')
Paths = jp.JClass('java.nio.file.Paths')

morphology = TurkishMorphology.createWithDefaults()

modelPath = Paths.get('ner-model')

PerceptronNer = jp.JClass('zemberek.ner.PerceptronNer')

ner = PerceptronNer.loadModel(modelPath, morphology)


def get_named_entities(input_str):

    result = ner.findNamedEntities(input_str)
    named_entities_list = result.getNamedEntities()

    named_entities_str = ""
    for item in named_entities_list:
        named_entities_str += item.toString() + " "
    return named_entities_str


jp.shutdownJVM()
