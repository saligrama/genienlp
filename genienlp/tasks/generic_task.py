#
# Copyright (c) 2019, The Board of Trustees of the Leland Stanford Junior University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from collections import OrderedDict

from ..data_utils.example import Example
from . import generic_dataset
from .almond_task import BaseAlmondTask
from .base_task import BaseTask
from .generic_dataset import CrossNERDataset, OODDataset
from .registry import register_task


@register_task('multi30k')
class Multi30K(BaseTask):
    @property
    def metrics(self):
        return ['bleu', 'em', 'nem', 'nf1']

    def get_splits(self, root, **kwargs):
        src, trg = ['.' + x for x in self.name.split('.')[1:]]
        return generic_dataset.Multi30k.splits(exts=(src, trg), root=root, **kwargs)


@register_task('iwslt')
class IWSLT(BaseTask):
    @property
    def metrics(self):
        return ['bleu', 'em', 'nem', 'nf1']

    def get_splits(self, root, **kwargs):
        src, trg = ['.' + x for x in self.name.split('.')[1:]]
        return generic_dataset.IWSLT.splits(exts=(src, trg), root=root, **kwargs)


@register_task('squad')
class SQuAD(BaseTask):
    @property
    def metrics(self):
        return ['nf1', 'em', 'nem']

    def get_splits(self, root, **kwargs):
        return generic_dataset.SQuAD.splits(root=root, description=self.name, **kwargs)


@register_task('wikisql')
class WikiSQL(BaseTask):
    @property
    def metrics(self):
        return ['lfem', 'em', 'nem', 'nf1']

    def get_splits(self, root, **kwargs):
        return generic_dataset.WikiSQL.splits(root=root, query_as_question='query_as_question' in self.name, **kwargs)


@register_task('ontonotes')
class OntoNotesNER(BaseTask):
    def get_splits(self, root, **kwargs):
        split_task = self.name.split('.')
        _, _, subtask, nones, counting = split_task
        return generic_dataset.OntoNotesNER.splits(
            subtask=subtask, nones=True if nones == 'nones' else False, root=root, **kwargs
        )


@register_task('woz')
class WoZ(BaseTask):
    @property
    def metrics(self):
        return ['joint_goal_em', 'turn_request_em', 'turn_goal_em', 'avg_dialogue', 'em', 'nem', 'nf1']

    def get_splits(self, root, **kwargs):
        return generic_dataset.WOZ.splits(description=self.name, root=root, **kwargs)


@register_task('multinli')
class MultiNLI(BaseTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.MultiNLI.splits(description=self.name, root=root, **kwargs)


@register_task('srl')
class SRL(BaseTask):
    @property
    def metrics(self):
        return ['nf1', 'em', 'nem']

    def get_splits(self, root, **kwargs):
        return generic_dataset.SRL.splits(root=root, **kwargs)


@register_task('snli')
class SNLI(BaseTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.SNLI.splits(root=root, **kwargs)


@register_task('schema')
class WinogradSchema(BaseTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.WinogradSchema.splits(root=root, **kwargs)


class BaseSummarizationTask(BaseTask):
    @property
    def metrics(self):
        return ['avg_rouge', 'rouge1', 'rouge2', 'rougeL', 'em', 'nem', 'nf1']


@register_task('cnn')
class CNN(BaseSummarizationTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.CNN.splits(root=root, **kwargs)


@register_task('dailymail')
class DailyMail(BaseSummarizationTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.DailyMail.splits(root=root, **kwargs)


@register_task('cnn_dailymail')
class CNNDailyMail(BaseSummarizationTask):
    def get_splits(self, root, **kwargs):
        split_cnn = generic_dataset.CNN.splits(root=root, **kwargs)
        split_dm = generic_dataset.DailyMail.splits(root=root, **kwargs)
        for scnn, sdm in zip(split_cnn, split_dm):
            scnn.examples.extend(sdm)
        return split_cnn


@register_task('sst')
class SST(BaseTask):
    def get_splits(self, root, **kwargs):
        return generic_dataset.SST.splits(root=root, **kwargs)


@register_task('imdb')
class IMDB(BaseTask):
    def get_splits(self, root, **kwargs):
        kwargs['validation'] = None
        return generic_dataset.IMDb.splits(root=root, **kwargs)


@register_task('zre')
class ZRE(BaseTask):
    @property
    def metrics(self):
        return ['corpus_f1', 'precision', 'recall', 'em', 'nem', 'nf1']

    def get_splits(self, root, **kwargs):
        return generic_dataset.ZeroShotRE.splits(root=root, **kwargs)


@register_task('cross_ner')
class CrossNERTask(BaseAlmondTask):
    politics_labels = [
        'O',
        'B-country',
        'B-politician',
        'I-politician',
        'B-election',
        'I-election',
        'B-person',
        'I-person',
        'B-organisation',
        'I-organisation',
        'B-location',
        'B-misc',
        'I-location',
        'I-country',
        'I-misc',
        'B-politicalparty',
        'I-politicalparty',
        'B-event',
        'I-event',
    ]

    science_labels = [
        'O',
        'B-scientist',
        'I-scientist',
        'B-person',
        'I-person',
        'B-university',
        'I-university',
        'B-organisation',
        'I-organisation',
        'B-country',
        'I-country',
        'B-location',
        'I-location',
        'B-discipline',
        'I-discipline',
        'B-enzyme',
        'I-enzyme',
        'B-protein',
        'I-protein',
        'B-chemicalelement',
        'I-chemicalelement',
        'B-chemicalcompound',
        'I-chemicalcompound',
        'B-astronomicalobject',
        'I-astronomicalobject',
        'B-academicjournal',
        'I-academicjournal',
        'B-event',
        'I-event',
        'B-theory',
        'I-theory',
        'B-award',
        'I-award',
        'B-misc',
        'I-misc',
    ]

    music_labels = [
        'O',
        'B-musicgenre',
        'I-musicgenre',
        'B-song',
        'I-song',
        'B-band',
        'I-band',
        'B-album',
        'I-album',
        'B-musicalartist',
        'I-musicalartist',
        'B-musicalinstrument',
        'I-musicalinstrument',
        'B-award',
        'I-award',
        'B-event',
        'I-event',
        'B-country',
        'I-country',
        'B-location',
        'I-location',
        'B-organisation',
        'I-organisation',
        'B-person',
        'I-person',
        'B-misc',
        'I-misc',
    ]

    literature_labels = [
        'O',
        'B-book',
        'I-book',
        'B-writer',
        'I-writer',
        'B-award',
        'I-award',
        'B-poem',
        'I-poem',
        'B-event',
        'I-event',
        'B-magazine',
        'I-magazine',
        'B-literarygenre',
        'I-literarygenre',
        'B-country',
        'I-country',
        'B-person',
        'I-person',
        'B-location',
        'I-location',
        'B-organisation',
        'I-organisation',
        'B-misc',
        'I-misc',
    ]

    ai_labels = [
        'O',
        'B-field',
        'I-field',
        'B-task',
        'I-task',
        'B-product',
        'I-product',
        'B-algorithm',
        'I-algorithm',
        'B-researcher',
        'I-researcher',
        'B-metrics',
        'I-metrics',
        'B-programlang',
        'I-programlang',
        'B-conference',
        'I-conference',
        'B-university',
        'I-university',
        'B-country',
        'I-country',
        'B-person',
        'I-person',
        'B-organisation',
        'I-organisation',
        'B-location',
        'I-location',
        'B-misc',
        'I-misc',
    ]

    news_labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    domain2labels = OrderedDict(
        {
            'politics': politics_labels,
            'science': science_labels,
            'music': music_labels,
            'literature': literature_labels,
            'ai': ai_labels,
            'news': news_labels,
        }
    )

    def __init__(self, name, args):
        self.all_labels = []
        # OrderedDict respect keys order
        for domain, labels in self.domain2labels.items():
            self.all_labels.extend(labels)
        self.label2id = {}
        for label in self.all_labels:
            if label not in self.label2id:
                self.label2id[label] = len(self.label2id)
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_labels = len(self.label2id)
        super().__init__(name, args)

    @property
    def metrics(self):
        return ['ner_f1', 'em', 'f1', 'pem']

    def _is_program_field(self, field_name):
        return field_name == 'answer'

    def utterance_field(self):
        return 'context'

    def _make_example(self, parts, dir_name=None, **kwargs):
        example_id = parts[0]
        context = ' '.join(parts[1])
        question = ''
        answer = ' '.join([str(self.label2id[label]) for label in parts[2]])

        return Example.from_raw(
            self.name + '/' + str(example_id), context, question, answer, preprocess=self.preprocess_field, lower=False
        )

    def get_splits(self, root, **kwargs):
        return CrossNERDataset.return_splits(name=self.name, path=root, make_example=self._make_example, **kwargs)


@register_task('ood_task')
class OODTask(BaseTask):
    def __init__(self, name, args):
        self.id2label = ['0', '1']
        self.num_labels = 2
        super().__init__(name, args)

    @property
    def metrics(self):
        return ['sc_f1', 'sc_precision', 'sc_recall']

    def get_splits(self, root, **kwargs):
        return OODDataset.splits(root=root, **kwargs)

@register_task('amazon_support_task')
class AmazonSupportTask(BaseTask):
    def __init__(self, name, args):
        self.id2label = [
            "",
            "order with release-date delivery",
            "learn how to use your send to kindle email address",
            "change your alexa device location",
            "the buyer-seller messaging service",
            "can't pair your fire tv remote",
            "amazon prime benefits",
            "fire tv support",
            "enter your packaging feedback",
            "install or update the kindle app on android",
            "revise payment",
            "gift orders",
            "vat rates",
            "deregister your device",
            "ordering",
            "alexa and alexa device faqs",
            "shipping speeds and charges",
            "kindle e-reader help",
            "place an order with exchange",
            "release day delivery",
            "order with release day delivery",
            "find a missing package that shows as delivered",
            "send a gift",
            "about restrictions on other delivery services",
            "request an a-to-z guarantee refund",
            "resolve a declined payment",
            "purchase from a  list",
            "amazon hub locker",
            "leave marketplace seller feedback",
            "wish lists",
            "create a child profile on your fire tablet",
            "how are shipping and delivery dates calculated?",
            "hello. what can we help you with?",
            "set your delivery instructions",
            "link your amazon and audible accounts",
            "about amazon pay gift card restrictions",
            "alexa doesn't understand or respond to your request",
            "redeem promotional codes",
            "community guidelines",
            "conditions of use&nbsp;&amp; sale",
            "improve your recommendations",
            "share your amazon prime benefits",
            "return a parcel at an amazon hub locker",
            "learn more about ads on kindle and fire tablet",
            "ordering from marketplace sellers",
            "terms and conditions for guaranteed delivery",
            "connect smart home devices to alexa",
            "support for echo plus",
            "listen to stations",
            "same-day (evening express) delivery",
            "end your amazon prime membership",
            "about goods and services tax (gst) for amazon business",
            "exchanges and replacements ",
            "add alexa contacts",
            "ordering from a third-party seller",
            "exchange offer - frequently asked questions",
            "tell us about a lower price",
            "subscribe &amp; save",
            "contact a marketplace seller ",
            "contact a marketplace seller",
            "echo device is having wi-fi issues",
            "shipping options",
            "about shipping restrictions",
            "easy monthly installments (emi)",
            "international delivery rates and times",
            "manage your browsing history",
            "where's my stuff?",
            "why do i have to pay shipping costs?",
            "exchange an item",
            " sign up for the amazon prime free trial",
            "review your alexa voice history",
            "what is two-step verification?",
            "undeliverable packages",
            "kindle fire doesn't charge",
            "kindle content help",
            "release day delivery terms &amp; conditions",
            "what is autorip?",
            "alexa help videos",
            "return items you ordered",
            "guaranteed shipping speeds and delivery charges",
            "track your package",
            "alexa terms of use",
            "conditions of use",
            "payment &amp; pricing",
            "view your gift card balance",
            "reset your password",
            "exchange offer",
            "amazon packaging",
            "report your amazon device as lost or stolen",
            "am i eligible for prime gaming?",
            "amazon pantry",
            "prime student terms &amp; conditions",
            "a-to-z guarantee",
            "recommended for you",
            "delivery speeds for global store items",
            "about import fees deposit",
            "download the alexa app",
            "alexa devices help",
            "communications from amazon.co.uk",
            "issues redeeming your online access code",
            "learn about using alexa on your fire tablet",
            "can't screen mirror on fire tv devices",
            "shipping and delivery",
            "items eligible for pick-up point delivery",
            "100% purchase protection",
            "cancel amazon music unlimited subscription",
            "refunds",
            "amazon pay gift cards",
            "fire tablet help",
            "frequently asked questions - topping up your balance",
            "what is alexa calling and messaging?",
            "unknown charges",
            "pair your phone or bluetooth speaker to your echo device",
            "play music with alexa using your voice",
            "authorizations",
            "ordering restrictions",
            "what is amazon music prime?",
            "amazon.in returns policy",
            "create your wish list",
            "things to try",
            "order a subscribe &amp; save subscription",
            "find an amazon hub locker",
            "money in amazon pay balance- faqs",
            "international shipping",
            "restrictions on international delivery on marketplace orders",
            "prime eligible items",
            "terms and conditions - amazon pay balance: money",
            "restart your kindle e-reader",
            "set up your fire tv",
            "returns and refunds",
            "update the wi-fi settings for your echo device",
            "about vat refunds for items to be exported outside the uk",
            "vat invoices",
            "fix a blank tv screen on fire tv devices",
            "marketplace returns and refunds",
            "track your return",
            "searching and browsing for items",
            "return a gift",
            "payment methods",
            "multi-room music does not play with alexa",
            "how are dispatch and delivery dates calculated?",
            "promotions and membership programmes",
            "buy add-on items",
            "amazon prime terms and conditions",
            "carrier contact information",
            "cancel items or orders",
            "terms and conditions: amazon pay gift cards issued by qwikcilver",
            "accessibility features for fire tv",
            "about amazon same-day grocery",
            "how do household profiles work on alexa devices?",
            "buy now ordering",
            "security and privacy",
            "cancel a request for a-to-z guarantee refund",
            "archive an order",
            "about returning items that contain hazardous materials",
            "what is x-ray for movies &amp; tv shows?",
            "download and install your digital software or video game",
            "about pay on delivery",
            "shopping with alexa",
            "change your account settings",
            "amazonglobal export countries and regions",
            "how to use your prime video watchlist",
            "amazon prime",
            "manage payment methods",
            "redeem a gift card",
            "recycle electrical or electronic equipment (weee)",
            "returns",
            "gift wrap",
            "learn about kindle unlimited",
            "manufacturer contact details and after sales service",
            "unsubscribe from amazon marketing",
            "amazon.in replacement policy",
            "about the message centre",
            "tax on items sold by sellers on amazon.ca",
            "sign up for text trace",
            "troubleshoot wi-fi connection issues with your fire tv",
            "can't connect to wi-fi",
            "set up your surprise spoiler settings",
            "change your subscription email preferences",
            "about packaging programmes",
            "kindle content isn't showing in your library",
            "change your buy now settings",
            "print an invoice",
            "find an amazon locker in your area",
            "amazon music",
            "alexa app is slow or unresponsive",
            "the amazon prime membership fee",
            "amazon prime shipping benefits",
            "prime student",
            "amazon appstore",
            "comments",
            "packaging programs",
            "amazon prime&nbsp;terms",
            "pricing for amazon global store products",
            "spilled or damaged amazon fresh items",
            "support options &amp; contact us",
            "exchange offer - terms & conditions",
            "install or uninstall amazon assistant",
            "link other music streaming services to alexa",
            "identifying whether an email",
            "view your a-to-z refund claim status",
            "use your mobile device like a fire tv remote",
            "disable smart home devices from alexa",
            "recommendations",
            "edit your reviews",
            "track by sms",
            "organise a return collection",
            "availability alerts",
            "manage bank account for refunds",
            "free delivery by amazon",
            "change your order information",
            "amazon.in refund policy",
            "one-day delivery and premium delivery",
            "get started with your business account",
            "payment verification for digital orders",
            "turn alexa skills on or off with your voice",
            "alexa features help",
            "proof of identity requirement",
            "amazon.in privacy notice",
            "we're sorry.",
            "resolve wi-fi connection issues on your kindle e-reader",
            "about free delivery on orders dispatched by amazon",
            "delivery guarantees ",
            "recharging a prepaid mobile number",
            "availability estimate definitions",
            "general shipping information",
            "payment",
            "which fire tv device do i have?",
            "managing your account",
            "order with prime free one-day delivery",
            "appeal a denied a-to-z guarantee refund",
            "customer reviews",
            "report a phishing email ",
            "delivery of large or bulky items",
            "check your echo device's software version",
            "amazon certified frustration-free packaging",
            "join prime student",
            "check the status of your refund",
            "find a missing parcel that shows as delivered",
            "add or remove content in fire for kids profiles",
            "request the closure of your account and the deletion of your personal information",
            "find a list",
            "submission of business proposals",
            "echo device is having bluetooth issues",
            "verify your email for a new account",
            "contact a third-party seller"
        ]
        self.num_labels = len(self.id2label)
        super().__init__(name, args)

    @property
    def metrics(self):
        return ['em']

    def get_splits(self, root, **kwargs):
        return AmazonSupportDataset.splits(root=root, **kwargs)