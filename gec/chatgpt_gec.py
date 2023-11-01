import os
import sys
import multiprocessing
import argparse
import re

from tqdm import tqdm
import openai

from sentence_segmentation import load_data, load_data_alignment

API_KEY = ""
base_url = "https://api.openai.com/v1/completions"
openai.api_key = API_KEY

OUTPUT_DIR = 'chatgpt/outputs'

examples_train = {
    'qalb14': [
        ('بسم الله  والصلاة والسلام علي رسول الله  وبعد : حقيقة  أنا من محبي البرنامج الحية لأستمع الي بعض الحلقات الماضية اللواتى لم أشاهدهن مثل الاتجاه المعاكس . وحلقة أكثر من رأي  ولكن عندما أشاهدها علي نت يكون فيها انقطاع الصوة  أعتذر للكم أن تعالجه  وشكرا موصول الكم ', 'بسم الله - والصلاة والسلام على رسول الله - وبعد : حقيقة ، أنا من محبي البرامج الحية ، أستمع إلى بعض الحلقات الماضية التي لم أشاهدها مثل الإتجاه المعاكس ، وحلقة أكثر من رأي ، ولكن عندما أشاهدها على النت يكون فيها انقطاع الصوت ، أقترح عليكم أن تعالجوه ، وشكري موصول إليكم .'),
        ('الله مبارك للقاعدة  كم بركتا على ابراهيم والي ابراهيم . واتمنا من قناة الجزيرة ان لاتنقلب لاجلي ارضاء السعودية . وكم شهدنا مراسل الجزيرة هذا العام بث على المباشر ايام العيد من مكة المكرمة . وهذا دليل انقلاب الجزيرة  في هاذه الايام نرها تلعب على اعصاب المسلمين وخاصة انصار القاعدة واحبائها  وهم والحمد لله بي الملاين . يا جزيرة  الحق ينتصر فلاتنقلبي . وعيد سعيد لكل مسلم يحب الله ','الله مبارك للقاعدة . كم بركة على إبراهيم وإلى إبراهيم ! وأتمنى من قناة الجزيرة أن لا تنقلب لأجل إرضاء السعودية . وكم شاهدنا مراسل الجزيرة هذا العام يبث على المباشر أيام العيد من مكة المكرمة ! وهذا دليل انقلاب الجزيرة ، في هذه الأيام نراها تلعب على أعصاب المسلمين وخاصة أنصار القاعدة وأحبائها ، وهم والحمد لله بالملايين . يا جزيرة ، الحق ينتصر فلا تنقلبي . وعيد سعيد لكل مسلم يحب الله .'),
        ('من ضمن خبركم أعلاه باستخدام القوات السورية لقصف مدنيين . . . شو  هاكذب يا جماعة الخير  كلها اخبار كاذبة  و مقرفة  و ممله  أيضا الفوضى بتصحيح مسار بأي حال من الاحول فائلة  و بالتالي خسرت المعارضة تأيدها للشعب السوري  و ما بني على باطل فهو باطل ', 'من ضمن خبركم أعلاه باستخدام القوات السورية لقصف مدنيين . . . ما هذا الكذب يا جماعة الخير ! كلها أخبار كاذبة ، ومقرفة ، ومملة . أيضا الفوضى بتصحيح مسار بأي حال من الأحوال قائمة ، وبالتالي خسرت المعارضة تأييدها للشعب السوري ، وما بني على باطل فهو باطل .')
    ],
    'qalb15_l2': [
        ('ذھبت الى السوق الجديد في صوفيا في الصيف الماضي . ھناك رأيت كثيرين من اصدقائي . رأيت مريا تجلس في المقھى وتشرب كولا . رايت يلينا تشتري كتابا كبيرا . رأيت سيلفيا تحمل زھرات كثيرة . رأيت ايفان يتكلم مع رجلا لا أعرفه . ثم ركبت الباص وبعد عدد دقائق وصلت الى مركز المدينة . ھناك رايت ناس كثيرين يلعبون مع اولادھم او يجلسون في المطعم او يذھبون إلى السينما او يمشون الى النافورة الكبيرة ويتكلمون .', 'ذهبت إلى السوق الجديد في صوفيا في الصيف الماضي . هناك رأيت كثيرا من أصدقائي . رأيت ماريا تجلس في المقهى وتشرب كولا . رأيت يلينا تشتري كتابا كبيرا . رأيت سيلفيا تحمل زهرات كثيرة . رأيت إيفان يتكلم مع رجل لا أعرفه . ثم ركبت الباص وبعد عدة دقائق وصلت إلى مركز المدينة . هناك رأيت ناسا كثيرين يلعبون مع أولادهم أو يجلسون في المطعم أو يذهبون إلى السينما أو يمشون إلى النافورة الكبيرة ويتكلمون .'),
        ('اختيارى تخصص الشرعي ادعو الله سبحانه وتعالى ان يجعل المسلمين والمسلمات فى ميزان الحسانته وان يرزقنا عما نافعا . لأني افضل على الشرعى لأكون في المستقبل داعيا إلى الله ومعلم أولاد المسلمين أمور الدين ليكون كلمة الله العليا . ليبقى الدين على الأرض سالما', 'اختياري تخصص الشريعة أدعو الله سبحانه وتعالى أن يجعل المسلمين والمسلمات في ميزان الحسنات وأن يرزقنا عملا نافعا ؛ لأني أفضل علم الشريعة لأكون في المستقبل داعيا إلى الله ومعلم أولاد المسلمين أمور الدين لتكون كلمة الله العليا . ليبقى الدين على الأرض سالما'),
        ('ستة اشھور بدآ علي يدرس في جامعة يوطا . كان يدرس درساة السرق الا سط . حصل على الشھادة في العلوم السياسية . ودرس ھناك متى رجعت سوزان إلى الامريلا مع زوجه . يفكر بھا فى غالب الاحيان لكن لا تكن تفكر به . كان قلبه بعدا من علي . كان اسمھا ويداد . درست العلوم السياسية في الجامعة يوط . بدأت تدرس مع الدكتر المشھور ، علي . كان يدر سھا وكانت يدرس بدون رسائل ، بدأا يتراسلان . يعرفھا و يحبھا . قالت : تزوجنى . قالت : ھم ا . انا مستعدة . كان مستعدا ايض ا . و زوجا . سكنوا في يوطا . لكن . . .', 'ستة أشهر بدأ علي يدرس في جامعة يوتا ، كان يدرس دراسات الشرق الأوسط . حصل على الشهادة في العلوم السياسية ، ودرس هناك متى رجعت سوزان إلى أمريكا مع زوجها . يفكر بها في غالب الأحيان لكن لم تكن تفكر به ، كان قلبها بعيدا عن علي . كان اسمها وداد . درست العلوم السياسية في جامعة يوتا . بدأت تدرس مع الدكتور المشهور ، علي . كان يدرس لها وكانت تدرس بدون رسائل . بدآ يتراسلان . يعرفها ويحبها . قالت : تزوجني . قالت : هيا ، أنا مستعدة . كان مستعدا أيضا . وتزوجا ، وسكنا في يوتا . لكن . . .')
    ],
    'zaebuc': [
        ('اصبح العديد من الناس يستخدم وسائل التواصل الاجتماعي بشكل مفرط فمنهم من يستخدمها دون هدف و منهم من يستخدمها للتجارة و لكن لنتحدث عن اللذين يستخدمونها دون هدف لما ذلك , هذا فقط لتضييع وقت الفراغ في بعض الامور و منهم من يتعارف على ثقافات دول اخرى لكن ان تركنا ذلك قليلا سنكتشف ان هناك الكثير من الاشياء تحدث فالواقع دون ان نلاحظها , بسبب مواقع التواصل الاجتماعي لم تعد بعض العائلات تجلس مع بعضها لم يعد احد يعرف الاخر ولكن من يمكنه ان يترك هذا الادمان', 'أصبح العديد من الناس يستخدم وسائل التواصل الاجتماعي بشكل مفرط فمنهم من يستخدمها دون هدف ومنهم من يستخدمها للتجارة ولكن لنتحدث عن الذين يستخدمونها دون هدف لما ذلك ? هذا فقط لتضييع وقت الفراغ في بعض الأمور ومنهم من يتعارف على ثقافات دول أخرى لكن إن تركنا ذلك قليلا سنكتشف أن هناك الكثير من الأشياء تحدث في الواقع دون أن نلاحظها  بسبب مواقع التواصل الاجتماعي لم تعد بعض العائلات تجلس مع بعضها لم يعد أحد يعرف الآخر ولكن من يمكنه أن يترك هذا الإدمان'),
        ('. يختلف آراء الجميع عن التواصل الإجتماعي بسبب كثرة تآثيراته السلبية والإيجابة فالتواصل الإجتماعي يفتح أبوب جديدة للتواصل بين أفراد المجتمع ولكن من الممكن أن تتجه هذه الأبواب إلى طرة مضرة من غير التوعية عن أخطارها .', ' تختلف آراء الجميع عن التواصل الاجتماعي بسبب كثرة تأثيراته السلبية والإيجابية فالتواصل الاجتماعي يفتح أبوابا جديدة للتواصل بين أفراد المجتمع ولكن من الممكن أن تتجه هذه الأبواب إلى طرق مضرة دون  التوعية بأخطارها .'),
        ('جميعنا , أو أغلبيتنا , نمتلك هاتفا و على هذه الهاتف هناك الكثير من مواقع التواصل الأجتماعي . هذه المواقع قد تجمعنا مع اقاربنا أو اصدقائنا أو الكثير من الناس حول العالم . و لكنها قد تلوث افكارنا و نزرع كره النفس , لكن يجب أن نتذكر أننا نحن الذين نمسك هذه الهواتف و نشعر بما نريد أن نشعر به . وسائل التواصل الإجتماعي احيانا تزيد الوعي عن الكوارث التي تحدث حول العالم و لكنها ايضا تسبب الكوارث يجب أن نحافظ على امننا ولا نعطي الجميع معلومات عن حياتنا لكي نصيح مجتمع سليم علينا الحفاظ على امانتنا و على أمانات الغير .', 'جميعنا  أو أغلبيتنا  نمتلك هاتفا وعلى هذا الهاتف هناك الكثير من مواقع التواصل الاجتماعي . هذه المواقع قد تجمعنا مع أقاربنا أو أصدقائنا أو الكثير من الناس حول العالم , ولكنها قد تلوث أفكارنا وتزرع كره النفس . لكن يجب أن نتذكر أننا نحن الذين نمسك هذه الهواتف ونشعر بما نريد أن نشعر به . وسائل التواصل الاجتماعي أحيانا تزيد الوعي عن الكوارث التي تحدث حول العالم ولكنها أيضا تسبب الكوارث يجب أن نحافظ على أمننا ولا نعطي الجميع معلومات عن حياتنا لكي نصبح مجتمعا سليما علينا الحفاظ على أمانتنا وعلى أمانات الغير .')
    ]
}
examples_train['qalb15_l1'] = examples_train['qalb15_l2']

def choose_n_examples(train_set):
    counter = {}
    for i, sent in enumerate(train_set['tags']):
        tags = [tag for tag in sent if tag != 'UC' and '_P' not in tag]
        tags_unique = set(tags)
        counter.setdefault(len(tags_unique), []).append((i, tuple(sorted(tags_unique))))
    pass


def prompt_template_n_shot(n, sent, examples_train):
    prompt_prelim = ('You are an Arabic grammatical error correction '
                     'tool that can identify and correct grammatical errors in a text.')

    prompt_instruct = ('Please identify and correct any spelling and grammar mistakes in the following '
                       'sentence indicated by <input> ERROR </input> tag. You need to comprehend the '
                       'sentence as a whole before gradually identifying and correcting any errors '
                       'while keeping the original sentence structure unchanged as much as possible. '
                       'Afterward, output the corrected version directly without any explanations. ')
    if n:
        prompt_prelim += (' We offer some examples labeled with the tag <input> SRC </input>,'
                          'representing original sentences that may contain grammatical errors. '
                          'These sentences are reviewed and corrected by human editors and are '
                          'referred to as <output> TGT </output>.')
        examples = ' Here are some in-context examples:\n'
        for i in range(n):
            src, tgt = examples_train[i]
            examples += f'({i + 1}), <input> {src} </input>: <output> {tgt} </output>.\n'
        examples += 'Please feel free to refer to these examples.\n'
        prompt_instruct += examples

    prompt_instruct += 'Remember to format your corrected output results with the tag <output> Your Corrected Version </output>.'

    messages = [{'role': 'system', 'content': prompt_prelim}]
    messages.append({'role': 'user', 'content': f'{prompt_instruct} Please start: <input> {sent} </input>'})
    return messages


def chatgpt_predict_multiproc(arguments):
    sent_split, sent_id, dataset_name, split, output_dir, n_shot, examples = arguments
    sent = ' '.join(sent_split)
    messages = prompt_template_n_shot(n_shot, sent, examples)
    try:
        chat = openai.ChatCompletion.create(
            model='gpt-3.5-turbo', messages=messages)
        pred = chat.choices[0].message.content
    except:
        pred = 'ERROR'

    output_dir = os.path.join(output_dir, dataset_name, split)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f'{sent_id}.txt'), 'w') as f:
        print(pred, file=f)

data_align = {}
load_data_alignment(data_align, name='qalb14', directory='data/alignment',
                    prefix='qalb14_', suffix='.areta+.txt', all_pairs=True)
load_data_alignment(data_align, name='qalb15_l2', directory='data/alignment',
                        prefix='qalb15_', suffix='.areta+.txt',
                        splits=['train', 'L2-test', 'dev'], all_pairs=True)
load_data_alignment(data_align, name='zaebuc', directory='data/alignment',
                        prefix='zaebuc_', suffix='.areta+.txt', all_pairs=True)

data = {}
load_data(data, name='qalb14', directory='data/chatgpt/qalb14',
              prefix='QALB-2014-L1-', suffix='.no_ids.clean.dediac',
              splits=['Train', 'Test', 'Dev'])
load_data(data, name='qalb15_l1', directory='data/chatgpt/qalb15',
              prefix='QALB-2015-L1-', suffix='.no_ids.dediac',
              splits=['Train', 'Test', 'Dev'])
load_data(data, name='qalb15_l2', directory='data/chatgpt/qalb15',
              prefix='QALB-2015-L2-', suffix='.no_ids.dediac',
              splits=['Train', 'Test', 'Dev'])
load_data(data, name='zaebuc', directory='data/chatgpt/zaebuc',
        suffix='.pnx.tok.dediac',
        src='sent.raw')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=100000,
                        type=int, help="Number of samples to take from the beginning of each set.")
    parser.add_argument("-n_shot", default=3,
                        type=int, help="What prompting scenario to use: zeros-hot, 3-shot, etc.")
    parser.add_argument("-datasets", nargs='+', default=['qalb14', 'qalb15_l1', 'qalb15_l2', 'zaebuc'],
                        type=str, help="Path of the directory where the sheets are.")
    parser.add_argument("-split", default='dev',
                        type=str, help="Which split to use for all datasets.")
    parser.add_argument("-output_dir", default=os.path.join(OUTPUT_DIR, 'chatgpt_output_0'),
                        type=str, help="Output directory.")
    parser.add_argument("-multiproc", default=False, action='store_true',
                        help="Whether or not to use multiprocessing.")
    args = parser.parse_args()

    DATASETS = [
        ('qalb14', args.split, 4),
        ('zaebuc', args.split, 6),
        ('qalb15_l2', args.split, 6),
        ('qalb15_l1', args.split, 6)
    ]

    output_dir = args.output_dir
    if os.path.exists(args.output_dir):
        resp = input('This directory already exists; overwrite [o], or new [n]: ')
        if resp == 'n':
            dirs = os.listdir(OUTPUT_DIR)
            dir_name = re.match(r'(.*)_\d+$', args.output_dir).group(1).split('/')[-1]
            max_int = max(int(re.match(f'{dir_name}_(\d+)$', directory).group(1))
                          for directory in dirs)

            output_dir = os.path.join(OUTPUT_DIR, f'{dir_name}_{max_int + 1}')
            os.makedirs(output_dir)
        elif resp == 'o':
            output_dir = args.output_dir
        else:
            raise NotImplementedError

    predictions = {}
    for dataset_name, split, n_cpu in DATASETS:
        if dataset_name not in args.datasets:
            continue

        examples_train_ = examples_train[dataset_name]
        data_ = [(sent, i, dataset_name, split, output_dir, args.n_shot, examples_train_)
                 for i, sent in enumerate(data[dataset_name][split]['src'][:args.n])]

        if args.multiproc:
            with multiprocessing.Pool(n_cpu) as p:
                preds = list(tqdm(p.imap(chatgpt_predict_multiproc, data_),
                                    total=len(data_), smoothing=0.2))
        else:
            preds = [chatgpt_predict_multiproc(sent_info) for sent_info in data_]

        for pred in preds:
            predictions.setdefault(dataset_name, {}).setdefault(
                split, []).append(pred)
