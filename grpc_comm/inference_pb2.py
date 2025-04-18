# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: inference.proto
# Protobuf Python Version: 5.27.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    27,
    2,
    '',
    'inference.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0finference.proto\x12\x0bspeculative\"o\n\x10LoadModelRequest\x12\x12\n\nmodel_path\x18\x01 \x01(\t\x12\x13\n\x0bn_positions\x18\x02 \x01(\r\x12\x12\n\nbatch_size\x18\x03 \x01(\r\x12\x11\n\ttp_degree\x18\x04 \x01(\r\x12\x0b\n\x03\x61mp\x18\x05 \x01(\t\"5\n\x11LoadModelResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\"<\n\x13StartSessionRequest\x12\x12\n\nsession_id\x18\x01 \x01(\t\x12\x11\n\tinput_ids\x18\x02 \x03(\r\"L\n\x14StartSessionResponse\x12\x12\n\nsession_id\x18\x01 \x01(\t\x12\x0f\n\x07success\x18\x02 \x01(\x08\x12\x0f\n\x07message\x18\x03 \x01(\t\"A\n\x14GenerateDraftRequest\x12\x13\n\x0bsession_ids\x18\x01 \x03(\t\x12\x14\n\x0c\x64raft_length\x18\x02 \x01(\r\"\xa2\x01\n\x15GenerateDraftResponse\x12?\n\x07outputs\x18\x01 \x03(\x0b\x32..speculative.GenerateDraftResponse.DraftOutput\x1aH\n\x0b\x44raftOutput\x12\x12\n\nsession_id\x18\x01 \x01(\t\x12\x0e\n\x06tokens\x18\x02 \x03(\r\x12\x15\n\rprobabilities\x18\x03 \x03(\x02\"Z\n\x19UpdateDraftContextRequest\x12\x12\n\nsession_id\x18\x01 \x01(\t\x12\x16\n\x0e\x61\x63\x63\x65pted_count\x18\x02 \x01(\r\x12\x11\n\tnew_token\x18\x03 \x01(\r\">\n\x1aUpdateDraftContextResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\"M\n\x11\x43heckTokenRequest\x12\x12\n\nsession_id\x18\x01 \x01(\t\x12\x10\n\x08token_id\x18\x02 \x01(\r\x12\x12\n\ndraft_prob\x18\x03 \x01(\x02\")\n\x12\x43heckTokenResponse\x12\x13\n\x0btarget_prob\x18\x01 \x01(\x02\":\n\x12\x41ppendTokenRequest\x12\x12\n\nsession_id\x18\x01 \x01(\t\x12\x10\n\x08token_id\x18\x02 \x01(\r\"&\n\x13\x41ppendTokenResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08\"G\n\x15GenerateTargetRequest\x12\x12\n\nsession_id\x18\x01 \x01(\t\x12\x1a\n\x12\x64raft_distribution\x18\x02 \x03(\x02\"*\n\x16GenerateTargetResponse\x12\x10\n\x08token_id\x18\x01 \x01(\r2\xee\x02\n\x0c\x44raftService\x12J\n\tLoadModel\x12\x1d.speculative.LoadModelRequest\x1a\x1e.speculative.LoadModelResponse\x12S\n\x0cStartSession\x12 .speculative.StartSessionRequest\x1a!.speculative.StartSessionResponse\x12V\n\rGenerateDraft\x12!.speculative.GenerateDraftRequest\x1a\".speculative.GenerateDraftResponse\x12\x65\n\x12UpdateDraftContext\x12&.speculative.UpdateDraftContextRequest\x1a\'.speculative.UpdateDraftContextResponse2\xbc\x03\n\rTargetService\x12J\n\tLoadModel\x12\x1d.speculative.LoadModelRequest\x1a\x1e.speculative.LoadModelResponse\x12S\n\x0cStartSession\x12 .speculative.StartSessionRequest\x1a!.speculative.StartSessionResponse\x12X\n\x15\x43heckTokenProbability\x12\x1e.speculative.CheckTokenRequest\x1a\x1f.speculative.CheckTokenResponse\x12P\n\x0b\x41ppendToken\x12\x1f.speculative.AppendTokenRequest\x1a .speculative.AppendTokenResponse\x12^\n\x13GenerateTargetToken\x12\".speculative.GenerateTargetRequest\x1a#.speculative.GenerateTargetResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'inference_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_LOADMODELREQUEST']._serialized_start=32
  _globals['_LOADMODELREQUEST']._serialized_end=143
  _globals['_LOADMODELRESPONSE']._serialized_start=145
  _globals['_LOADMODELRESPONSE']._serialized_end=198
  _globals['_STARTSESSIONREQUEST']._serialized_start=200
  _globals['_STARTSESSIONREQUEST']._serialized_end=260
  _globals['_STARTSESSIONRESPONSE']._serialized_start=262
  _globals['_STARTSESSIONRESPONSE']._serialized_end=338
  _globals['_GENERATEDRAFTREQUEST']._serialized_start=340
  _globals['_GENERATEDRAFTREQUEST']._serialized_end=405
  _globals['_GENERATEDRAFTRESPONSE']._serialized_start=408
  _globals['_GENERATEDRAFTRESPONSE']._serialized_end=570
  _globals['_GENERATEDRAFTRESPONSE_DRAFTOUTPUT']._serialized_start=498
  _globals['_GENERATEDRAFTRESPONSE_DRAFTOUTPUT']._serialized_end=570
  _globals['_UPDATEDRAFTCONTEXTREQUEST']._serialized_start=572
  _globals['_UPDATEDRAFTCONTEXTREQUEST']._serialized_end=662
  _globals['_UPDATEDRAFTCONTEXTRESPONSE']._serialized_start=664
  _globals['_UPDATEDRAFTCONTEXTRESPONSE']._serialized_end=726
  _globals['_CHECKTOKENREQUEST']._serialized_start=728
  _globals['_CHECKTOKENREQUEST']._serialized_end=805
  _globals['_CHECKTOKENRESPONSE']._serialized_start=807
  _globals['_CHECKTOKENRESPONSE']._serialized_end=848
  _globals['_APPENDTOKENREQUEST']._serialized_start=850
  _globals['_APPENDTOKENREQUEST']._serialized_end=908
  _globals['_APPENDTOKENRESPONSE']._serialized_start=910
  _globals['_APPENDTOKENRESPONSE']._serialized_end=948
  _globals['_GENERATETARGETREQUEST']._serialized_start=950
  _globals['_GENERATETARGETREQUEST']._serialized_end=1021
  _globals['_GENERATETARGETRESPONSE']._serialized_start=1023
  _globals['_GENERATETARGETRESPONSE']._serialized_end=1065
  _globals['_DRAFTSERVICE']._serialized_start=1068
  _globals['_DRAFTSERVICE']._serialized_end=1434
  _globals['_TARGETSERVICE']._serialized_start=1437
  _globals['_TARGETSERVICE']._serialized_end=1881
# @@protoc_insertion_point(module_scope)
