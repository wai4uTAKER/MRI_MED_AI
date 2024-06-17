import json
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework import status
from rest_framework.views import APIView
from nnmodel import models


# from nnmodel.nn.defaultModels import DefalutModels
from nnmodel.apps import NNmodelConfig
from nnmodel.nn.loaders.img_loader import defaultImgLoader
import cv2 as cv

"""SERIALIZERS"""
from rest_framework.serializers import Serializer, ModelSerializer
import rest_framework.serializers as ser


class PredictAllSerializer(Serializer):
    file_path = ser.CharField()
    projection_type = ser.CharField()
    id = ser.IntegerField()


class UZISegmentationForm(ModelSerializer):
  def __init__(self, instance=None, data=..., **kwargs):
    super().__init__(instance, data, **kwargs)

  nodule_type = ser.IntegerField(
    min_value=1,
    max_value=5,
    default=1,
    allow_null=True
  )

  nodule_2_3 = ser.FloatField(
    default=0,
    min_value=0,
    max_value=1
  )
  nodule_4 = ser.FloatField(
    default=0,
    min_value=0,
    max_value=1
  )
  nodule_5 = ser.FloatField(
    default=0,
    min_value=0,
    max_value=1
  )

  nodule_width = ser.FloatField(
    default=1,
    min_value=0,
  )
  nodule_height = ser.FloatField(
    default=1,
    min_value=0,
  )
  nodule_length = ser.FloatField(
    default=1,
    min_value=0,
  )

  class Meta:
    model = models.SegmentationData
    exclude = ['details', 'original_image']

  def update(self, instance, validated_data):
    return super().update(instance, validated_data)



def segmetationDataForm(nn_class):
  data = {
    "nodule_type": nn_class.argmax() + 3,
    "nodule_2_3": nn_class[0],
    "nodule_4": nn_class[1],
    "nodule_5": nn_class[2],
  }
  ser = UZISegmentationForm(data=data)
  ser.is_valid(raise_exception=True)
  return ser.validated_data


"""APIVIEWS"""
from nnmodel.nn.loaders.img_loader import defaultImgLoader
from nnmodel.apps import NNmodelConfig
import cv2 as cv


def createUziSegmentGroup(details, uzi_image_id, form=segmetationDataForm):
    return models.UZISegmentGroupInfo(
        details=form(details),
        is_ai=True,
        original_image_id=uzi_image_id
    )


class PredictAll(APIView):
    serializer_class = PredictAllSerializer

    def post(self, request: Request, *args, **kwargs):
    
        ser = PredictAllSerializer(data=request.data)
        ser.is_valid(raise_exception=True)

        adenoma_count = self.predict(
            ser.validated_data['file_path'],
            ser.validated_data['projection_type'],
            ser.validated_data['id']
        )
        return Response({'adenoma_count': adenoma_count})
        

    def predict(self, file_path: str, projection_type: str, id: int):
        projection_type = 'all'
        print(f'predictions, {projection_type=} {file_path=}')
        nn_cls = NNmodelConfig.DefalutModels['C']['all'] 
        projection_type= 'cross'
        nn_seg = NNmodelConfig.DefalutModels['S'][projection_type]
        projection_type = 'all'
        img = defaultImgLoader.load(file_path)
        seg_nodules = nn_seg.predict(img)
        ind, track = nn_cls.predict(nn_seg.rois, nn_seg.img_type)


        # print(f'predictions, {projection_type=} {file_path=}')
        # details = {}
        # for key,val in track.items():
        #     details[key] = createUziSegmentGroup(val, id)
        # print(f'predictions, {projection_type=} {file_path=}')
        # models.UZISegmentGroupInfo.objects.bulk_create(details.values())

        # print(f'predictions, {projection_type=} {file_path=}')
        # segments_data = []
        # for ni in ind:
        #     for idx, nj in ni.items():
        #         for nj2 in nj:
        #             pre_details = segmetationDataForm(nj2,True)
        #             segments_data.append(models.SegmentationData(
        #                 segment_group=details[idx], 
        #                 details=pre_details
        #             ))
        # print(f'predictions, {projection_type=} {file_path=}')            
        # models.SegmentationData.objects.bulk_create(segments_data)

        segments_points = []
        print(f'predictions, {projection_type=} {file_path=}')
        for k,seg in enumerate(seg_nodules):
            c,h = cv.findContours(seg[0],cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE | cv.CHAIN_APPROX_TC89_L1)
            if h is not None:
                counter = [c[i] for i in range(h.shape[1]) if h[0][i][3] == -1][0]
                # for i, pi in enumerate(counter):
                #     segments_points.append(models.SegmentationPoint(
                #         uid=i, segment=segments_data[k],
                #         x=pi[0,0],
                #         y=pi[0,1],
                #         z=seg[2]
                #     ))
        # print(f'ЗЗЗЗЗЗredictions, {projection_type=} {file_path=}')
        # models.SegmentationPoint.objects.bulk_create(segments_points, batch_size=2048)
        # models.OriginalImage.objects.filter(id=id).update(viewed_flag=True)
        print('predicted!')
        print(k)
        return k
