# source files of library
SET(CrossCheckRawdepth_SRCS
  CrossCheckRawdepth.cpp
)

PIP_ADD_EXECUTABLE(CrossCheckRawdepth ${CrossCheckRawdepth_SRCS})
TARGET_LINK_LIBRARIES(CrossCheckRawdepth PIPBase PIPInterOpCUDA PIPAlgorithms)
